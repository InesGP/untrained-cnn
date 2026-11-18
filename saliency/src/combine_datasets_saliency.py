"""
Combine saliency maps across several datasets (NKI + HBN by default) into ONE
map per model, per target.

Same idea as aggregate_saliency.py, but pooling the per-subject maps of
several dataset trees at once (saliency_affine_pretrained/<target>/nki/ and
.../hbn/) instead of one tree at a time. Everything downstream (regionalize_
saliency.py, average_uncnn_channels.py, visualize_paper_nilearn.py,
visualize_sanity_check.py) reads the combined output through the usual
filename conventions, so none of them need to know it came from 2 datasets.

Baseline mode (default)
-----------------------
For every (model, kind) pools every subject map
    <dataset-dir>/<SUBJECT>/{model}_{attribution_tag}{kind}_saliency.nii.gz
from ALL --dataset-dirs and writes
    <output-dir>/{model}_{attribution_tag}aggregate_{stat}_{kind}_saliency.nii.gz
for stat in --stats (mean, std; same population std (ddof=0) as
aggregate_saliency.py). This is a SUBJECT-weighted pool -- every subject counts
once, so the larger dataset contributes proportionally more -- not a mean of the
per-dataset means. Accumulated as streaming sum/sum-of-squares (float64) so
memory stays at a couple of volumes however many subjects there are. A
sidecar .json next to each output records how many subjects each dataset
contributed. Subject maps whose grid differs from the first one seen are
skipped and reported, same safety net as aggregate_saliency.py.

--sanity-check {arch,data,input}
--------------------------------
  arch : pools the per-subject randomized-model maps
         <dataset-dir>/<--arch-persubject-dirname>/<SUBJECT>/arch_..._{kind}_saliency.nii.gz
         (slurm/compute_sanity_checks_randomize_model.sh's output) across all
         datasets, i.e. what aggregate_saliency.py --sanity-check arch does for
         one dataset, but for several at once.
  data / input : these are single reference-subject maps (one per dataset, at
         <dataset-dir>/{data_,input_}...{kind}_saliency.nii.gz), so the combined
         map is the plain mean of the datasets' maps. A (model, kind) missing in
         ANY dataset is skipped with a warning rather than silently written from
         the remaining datasets only -- a "combined" map built from one dataset
         would be misleading.
Output filenames are exactly the single-dataset ones
(<output-dir>/arch_{model}_{variant}_{tag}{kind}_saliency.nii.gz, ...), which is
what visualize_paper_nilearn.py --sanity-check / visualize_sanity_check.py read.
--models is required (sanity filenames aren't auto-discoverable, see
aggregate_saliency.py's module docstring).

Usage:
    python combine_datasets_saliency.py \\
        --dataset-dirs saliency_affine_pretrained/age/nki saliency_affine_pretrained/age/hbn \\
        --output-dir saliency_affine_pretrained/age/nki_hbn/aggregate \\
        --models brainiac simclr --stats mean std --kinds signed abs

    python combine_datasets_saliency.py --sanity-check arch --models brainiac simclr \\
        --dataset-dirs ... --output-dir saliency_affine_pretrained/age/nki_hbn
"""

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aggregate_saliency import sanity_prefix  # noqa: E402  (nilearn-free; see its module docstring)

DEFAULT_MODELS = (
    "optim_uncnn_raw", "optim_uncnn_rank", "optim_uncnn_sobel", "brainiac", "swinbrain", "simclr",
    "anatcl_global", "anatcl_local", "fs_roi",
)
NON_SUBJECT_DIRS = ("aggregate", "aggregate_roi")


def subject_dirs_of(dataset_dir, exclude=(), exclude_subjects=()):
    """Per-subject subfolders of `dataset_dir` (skips aggregate*/ *_persubject/ trees, `exclude` paths and `exclude_subjects` ids)."""
    exclude = {Path(p).resolve() for p in exclude}
    return sorted(
        p for p in Path(dataset_dir).iterdir()
        if p.is_dir() and p.name not in NON_SUBJECT_DIRS and not p.name.endswith("_persubject")
        and p.name not in exclude_subjects and p.resolve() not in exclude
    )


def collect(dataset_dirs, filename, subdir=None, exclude=(), exclude_subjects=()):
    """[(dataset_name, subject_id, Path)] for every subject whose folder holds `filename`."""
    found = []
    for dataset_dir in map(Path, dataset_dirs):
        root = dataset_dir / subdir if subdir else dataset_dir
        if not root.is_dir():
            continue
        for subj in subject_dirs_of(root, exclude, set(exclude_subjects)):
            f = subj / filename
            if f.exists():
                found.append((dataset_dir.name, subj.name, f))
    return found


def pool_subjects(files, stats):
    """
    Streaming pooled mean/std over [(dataset, subject, Path)]. Returns
    ({stat: ndarray}, affine, per-dataset counts, skipped list), or
    (None, None, {}, skipped) if nothing usable.
    """
    total = total_sq = ref_shape = ref_affine = None
    counts, skipped, n = {}, [], 0
    for dataset, subject, path in files:
        img = nib.load(path)
        if ref_shape is None:
            ref_shape, ref_affine = img.shape, img.affine
        elif img.shape != ref_shape or not np.allclose(img.affine, ref_affine, atol=1e-3):
            skipped.append(f"{dataset}/{subject}")
            continue
        data = img.get_fdata(dtype=np.float32).astype(np.float64)
        if total is None:
            total, total_sq = np.zeros(ref_shape), np.zeros(ref_shape)
        total += data
        if "std" in stats:
            total_sq += data * data
        counts[dataset] = counts.get(dataset, 0) + 1
        n += 1
    if n == 0:
        return None, None, {}, skipped
    mean = total / n
    out = {}
    if "mean" in stats:
        out["mean"] = mean
    if "std" in stats:
        out["std"] = np.sqrt(np.maximum(total_sq / n - mean * mean, 0.0))
    return out, ref_affine, counts, skipped


def _save(arr, affine, path, sidecar):
    nib.save(nib.Nifti1Image(arr.astype(np.float32), affine), path)
    path.with_suffix("").with_suffix(".json").write_text(json.dumps(sidecar, indent=2))
    print(f"    -> saved {path}")


def run_pooled_job(job):
    """One (model, kind) pool; module-level so ProcessPoolExecutor can pickle it."""
    label, files, stats, out_paths, sidecar_base = job
    out, affine, counts, skipped = pool_subjects(files, stats)
    if out is None:
        return f"{label}: no usable maps -- skipped"
    if skipped:
        print(f"    {label}: skipped {len(skipped)} grid-mismatched map(s): {skipped[:5]}...")
    for stat, path in out_paths.items():
        _save(out[stat], affine, path, {**sidecar_base, "stat": stat, "n_subjects": sum(counts.values()), "n_per_dataset": counts})
    return f"{label}: pooled {sum(counts.values())} maps {counts}"


def run_reference_mean(dataset_dirs, filename, out_path, label, sidecar_base):
    """Mean of the single per-dataset reference maps; requires ALL datasets to have it."""
    paths = [(Path(d).name, Path(d) / filename) for d in dataset_dirs]
    missing = [name for name, p in paths if not p.exists()]
    if missing:
        return f"{label}: missing in {missing} -- skipped (won't write a 'combined' map from a subset of datasets)"
    imgs = [(name, nib.load(p)) for name, p in paths]
    ref = imgs[0][1]
    for name, img in imgs:
        if img.shape != ref.shape or not np.allclose(img.affine, ref.affine, atol=1e-3):
            return f"{label}: grid mismatch for {name} ({img.shape} vs {ref.shape}) -- skipped"
    mean = np.mean([img.get_fdata(dtype=np.float32) for _, img in imgs], axis=0)
    _save(mean, ref.affine, out_path, {**sidecar_base, "datasets": [name for name, _ in paths], "stat": "mean"})
    return f"{label}: averaged {len(imgs)} dataset reference maps"


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--dataset-dirs", nargs="+", type=Path, required=True,
                        help="Per-dataset trees to pool, e.g. saliency_affine_pretrained/age/nki .../hbn.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="*", default=list(DEFAULT_MODELS))
    parser.add_argument("--kinds", nargs="*", default=["signed", "abs"], choices=["signed", "abs"])
    parser.add_argument("--stats", nargs="*", default=["mean", "std"], choices=["mean", "std"],
                        help="Baseline mode only (sanity-check outputs are always the mean).")
    parser.add_argument("--attribution-tag", default="ig_reference_",
                        help="Attribution tag baked into the filenames ('' for gradient). Default: ig_reference_.")
    parser.add_argument("--sanity-check", choices=["arch", "data", "input"], default=None)
    parser.add_argument("--arch-variant", default="step01_head")
    parser.add_argument("--arch-persubject-dirname", default="arch_linear_persubject")
    parser.add_argument("--exclude-subjects", nargs="*", default=[],
                        help="Subject ids to leave out of every pool (e.g. the duplicated-T1w HBN subjects, see "
                        "compute_saliency_pretrained_heads.sh's EXCLUDE_SUBJECTS). Baseline/arch modes only.")
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel (model, kind) pools (baseline/arch modes).")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tag, results, jobs = args.attribution_tag, [], []

    if args.sanity_check in ("data", "input"):
        for model in args.models:
            for kind in args.kinds:
                prefix = sanity_prefix(model, args.sanity_check, args.arch_variant)
                name = f"{prefix}{tag}{kind}_saliency.nii.gz"
                results.append(run_reference_mean(
                    args.dataset_dirs, name, args.output_dir / name, f"{args.sanity_check} {model} {kind}",
                    {"sanity_check": args.sanity_check, "model": model},
                ))
    else:
        for model in args.models:
            for kind in args.kinds:
                if args.sanity_check == "arch":
                    prefix = sanity_prefix(model, "arch", args.arch_variant)
                    name = f"{prefix}{tag}{kind}_saliency.nii.gz"
                    files = collect(args.dataset_dirs, name, subdir=args.arch_persubject_dirname, exclude_subjects=args.exclude_subjects)
                    out_paths = {"mean": args.output_dir / name}
                    base = {"sanity_check": "arch", "randomize_model_variant": args.arch_variant, "model": model}
                    stats = ["mean"]
                else:
                    name = f"{model}_{tag}{kind}_saliency.nii.gz"
                    files = collect(args.dataset_dirs, name, exclude=(args.output_dir,), exclude_subjects=args.exclude_subjects)
                    out_paths = {
                        s: args.output_dir / f"{model}_{tag}aggregate_{s}_{kind}_saliency.nii.gz" for s in args.stats
                    }
                    base = {"model": model}
                    stats = list(args.stats)
                label = f"{args.sanity_check or 'baseline'} {model} {kind}"
                if not files:
                    results.append(f"{label}: no maps found in any dataset -- skipped")
                    continue
                if args.sanity_check == "arch":
                    # Same rule as data/input: a "combined" arch map must include every dataset.
                    lacking = sorted({d.name for d in args.dataset_dirs} - {d for d, _, _ in files})
                    if lacking:
                        results.append(f"{label}: no per-subject maps in {lacking} -- skipped (won't write a 'combined' map from a subset of datasets)")
                        continue
                jobs.append((label, files, stats, out_paths, {**base, "datasets": [d.name for d in args.dataset_dirs]}))

        if args.n_jobs > 1 and len(jobs) > 1:
            with ProcessPoolExecutor(max_workers=args.n_jobs) as pool:
                results += list(pool.map(run_pooled_job, jobs))
        else:
            results += [run_pooled_job(j) for j in jobs]

    print("\n=== Summary ===")
    for r in results:
        print(f"  {r}")
    if not any("pooled" in r or "averaged" in r for r in results):
        raise SystemExit("Nothing was written.")


if __name__ == "__main__":
    main()
