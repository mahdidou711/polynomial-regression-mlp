"""Entry point for exporting the tables and figures into results/.

Commands (none of them trains an MLP campaign; the test set is never read except for
the figures of ``--with-models``, after the freeze has been verified):

  python main.py verify [--with-models]
      Read-only. Static checks of the exported artefacts (naming, formats,
      manifest, consistency of the final evaluation), then
      regeneration in a temporary directory and byte-for-byte comparison.

  python main.py export [--with-models]
      Regenerates in a temporary directory, refuses any drift of the source tables, then
      writes into the repository only the derived artefacts that differ (figures, derived CSV,
      figure metadata, manifest). Source tables already present are never
      rewritten; missing ones are created, so the first export in a clean checkout builds
      results/ from the cache of trained networks. ``--with-models`` adds the figures that
      require three rebuilt networks whose weight fingerprints are verified.

  python main.py reproduce --out DIR [--no-models]
      Isolated, explicit reproduction: everything is written under DIR (outside the repository),
      then compared with the repository. Retrains no campaign; the run cache is read, never written.

  python main.py reproduce --full --out DIR
      FULL, isolated and explicit reproduction: 1D runners, stages B and C with training of the
      190 runs in the isolated store DIR/store, final evaluation, figures, tables, manifests.
      Safeguard: the test set is read only after checking that the retrained runs and the
      recomputed selection freeze are scientifically identical to the reference. The final
      comparison ignores run durations (not reproducible) and nothing is written to the repository.

A single BLAS thread is required: this file sets it before numpy is imported.
"""

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")  # before any numpy import

import argparse  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

from src.export import checks, pipeline  # noqa: E402
from src.export.layout import Layout, PROJECT_ROOT, REPO  # noqa: E402

MARK = {checks.PASSED: "PASSED ", checks.FAILED: "FAILED ", checks.WARNING: "WARNING",
        checks.NOT_RUN: "NOT RUN"}


def report(items) -> int:
    """Print the checks; return the number of failures."""
    failed = 0
    for item in items:
        print(f"[{MARK[item.status]}] {item.name}")
        for line in item.details[:12]:
            print(f"           - {line}")
        if len(item.details) > 12:
            print(f"           ... {len(item.details) - 12} more")
        failed += item.status == checks.FAILED
    return failed


def diff_check(name, diff) -> checks.Check:
    issues = ([f"missing from the repository: {p}" for p in diff["missing_in_second"]]
              + [f"missing from the regeneration: {p}" for p in diff["missing_in_first"]]
              + [f"different content: {p}" for p in diff["different"]])
    return checks.Check.from_issues(name, issues)


def cmd_verify(args) -> int:
    items = checks.run_static_checks(REPO)
    try:
        with tempfile.TemporaryDirectory(prefix="c0_verify_") as tmp:
            out = Layout(Path(tmp))
            info = pipeline.produce(out, REPO, with_models=args.with_models)
            sources = pipeline.compare(out, REPO, only=pipeline.source_paths())
            items.append(diff_check("derived from the source tables: 1D recomputed, MLP from the cache",
                                    sources))
            everything = pipeline.compare(out, REPO, exclude=pipeline.source_paths())
            items.append(diff_check("derived artefacts: byte-identical regeneration",
                                    everything))
            if args.with_models:
                items.append(checks.Check("figures that depend on networks: regenerated and compared",
                                          checks.PASSED if pipeline.is_clean(everything) else checks.FAILED))
            else:
                items.append(checks.Check(
                    "figures that depend on networks: regeneration", checks.NOT_RUN,
                    ["needs 3 verified rebuilds; rerun with --with-models"]))
    except (pipeline.ExportError, RuntimeError) as exc:
        items.append(checks.Check("regeneration", checks.FAILED, [str(exc)]))
    failed = report(items)
    print(f"verify: {'FAILED' if failed else 'OK'} ({failed} failed check(s))")
    return 1 if failed else 0


def cmd_export(args) -> int:
    with tempfile.TemporaryDirectory(prefix="c0_export_") as tmp:
        out = Layout(Path(tmp))
        try:
            info = pipeline.produce(out, REPO, with_models=args.with_models)
        except (pipeline.ExportError, RuntimeError) as exc:
            print(f"export refused: {exc}")
            return 2
        # Source tables already present are never rewritten; missing ones are created
        # (first export in a repository without results/).
        present = {p for p in pipeline.source_paths() if (REPO.root / p).exists()}
        drift = pipeline.compare(out, REPO, only=present)
        if not pipeline.is_clean(drift):
            report([diff_check("derived from the source tables: export refused, repository unchanged", drift)])
            return 2
        changed = pipeline.sync(out, REPO, exclude=present)
    print(f"export: {len(changed)} file(s) written or modified; models: {info['model_figures']}")
    for path in changed:
        print(f"  {path}")
    return 1 if report(checks.run_static_checks(REPO)) else 0


def cmd_reproduce(args) -> int:
    out_dir = Path(args.out).resolve()
    try:
        out_dir.relative_to(PROJECT_ROOT)
    except ValueError:
        pass
    else:
        print("reproduce refused: --out must be outside the repository")
        return 2
    if out_dir.exists() and any(out_dir.iterdir()):
        print(f"reproduce refused: {out_dir} is not empty")
        return 2
    if args.full and args.no_models:
        print("reproduce refused: --full and --no-models are incompatible")
        return 2
    out = Layout(out_dir)
    try:
        if args.full:
            info = pipeline.produce_full(out, REPO)
        else:
            info = pipeline.produce(out, REPO, with_models=not args.no_models)
    except (pipeline.ExportError, RuntimeError) as exc:
        print(f"reproduce refused: {exc}")
        return 2
    if args.full:
        from src.experiments.runner_2d_nominal import NOMINAL_STORE
        stores = checks.Check.from_issues(
            f"{info['runs_compared']} retrained runs: weights and metrics identical to the valid cache",
            pipeline.compare_stores(out.root / "store", NOMINAL_STORE))
        diff = diff_check(f"full reproduction ({out_dir}): scientific content identical to the repository",
                          pipeline.compare_scientific(out, REPO))
        items = [stores, diff, *checks.run_static_checks(out)]
    else:
        diff = pipeline.compare(out, REPO)
        items = [diff_check(f"isolated reproduction ({out_dir}) identical to the repository", diff),
                 *checks.run_static_checks(out)]
    failed = report(items)
    print(f"reproduce: {'FAILED' if failed else 'OK'} ; models: {info['model_figures']}")
    return 1 if failed else 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("verify", "export"):
        p = sub.add_parser(name)
        p.add_argument("--with-models", action="store_true",
                       help="also regenerate the figures that require 3 rebuilt networks")
    p = sub.add_parser("reproduce")
    p.add_argument("--out", required=True, help="empty directory, outside the repository")
    p.add_argument("--no-models", action="store_true")
    p.add_argument("--full", action="store_true",
                   help="full reproduction: retrain everything in DIR/store (explicit, isolated)")
    args = parser.parse_args(argv)
    return {"verify": cmd_verify, "export": cmd_export, "reproduce": cmd_reproduce}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
