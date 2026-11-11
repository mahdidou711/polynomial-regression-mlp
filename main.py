"""Point d'entree de l'export pour le rapport LaTeX (etape 8).

Commandes (toutes sans entrainement de campagne MLP ; la base de test n'est jamais lue
hors des figures de ``--with-models``, apres verification du gel, comme a l'etape 7) :

  python main.py verify [--with-models]
      Lecture seule. Controles statiques des artefacts versionnes (nomenclature, formats,
      tables .tex, apercu, manifeste, hygiene, coherence de l'evaluation finale), puis
      regeneration dans un repertoire temporaire et comparaison octet a octet.

  python main.py export [--with-models]
      Regenere dans un repertoire temporaire, refuse toute derive des tables sources, puis
      ecrit dans le depot uniquement les artefacts derives qui different (figures, tables .tex
      et CSV derives, metadonnees de figures, apercu, manifeste). Les tables sources ne sont
      jamais reecrites. ``--with-models`` ajoute les figures qui exigent trois reseaux
      reconstruits et verifies par l'empreinte de leurs poids.

  python main.py reproduce --out DIR [--no-models]
      Reproduction isolee et explicite : tout est ecrit sous DIR (hors du depot), puis
      comparee au depot. Ne reentraine aucune campagne ; le cache des runs est lu, jamais ecrit.

  python main.py reproduce --full --out DIR
      Reproduction INTEGRALE, isolee et explicite (plan 04, section 8, critere 1) : runners 1D,
      lots B et C avec entrainement des 190 runs dans le depot isole DIR/store, evaluation
      finale, figures, tables, manifestes. Garde-fou DEC-20 / R3 : la base de test n'est lue
      qu'apres avoir constate que les runs reentraines et le gel de selection recalcule sont
      scientifiquement identiques a ceux de la reference validee. La comparaison finale ignore
      les durees d'execution (non reproductibles) et rien n'est ecrit dans le depot.

Un seul thread BLAS est exige (D7-04) : ce fichier le fixe avant l'import de numpy.
"""

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")  # avant tout import de numpy

import argparse  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

from src.export import checks, pipeline  # noqa: E402
from src.export.layout import Layout, PROJECT_ROOT, REPO  # noqa: E402

MARK = {checks.PASSED: "PASSED ", checks.FAILED: "FAILED ", checks.WARNING: "WARNING",
        checks.NOT_RUN: "NOT RUN"}


def report(items) -> int:
    """Imprime les controles ; renvoie le nombre d'echecs."""
    failed = 0
    for item in items:
        print(f"[{MARK[item.status]}] {item.name}")
        for line in item.details[:12]:
            print(f"           - {line}")
        if len(item.details) > 12:
            print(f"           ... {len(item.details) - 12} autres")
        failed += item.status == checks.FAILED
    return failed


def diff_check(name, diff) -> checks.Check:
    issues = ([f"absent du depot : {p}" for p in diff["missing_in_second"]]
              + [f"absent de la regeneration : {p}" for p in diff["missing_in_first"]]
              + [f"contenu different : {p}" for p in diff["different"]])
    return checks.Check.from_issues(name, issues)


def cmd_verify(args) -> int:
    items = checks.run_static_checks(REPO)
    try:
        with tempfile.TemporaryDirectory(prefix="c0_verify_") as tmp:
            out = Layout(Path(tmp))
            info = pipeline.produce(out, REPO, with_models=args.with_models)
            sources = pipeline.compare(out, REPO, only=pipeline.source_paths())
            items.append(diff_check("derive des tables sources (AE4-04) : 1D recalcul, MLP depuis le cache",
                                    sources))
            everything = pipeline.compare(out, REPO, exclude=pipeline.source_paths())
            items.append(diff_check("artefacts derives : regeneration identique octet a octet",
                                    everything))
            if args.with_models:
                items.append(checks.Check("figures dependant des reseaux : regenerees et comparees",
                                          checks.PASSED if pipeline.is_clean(everything) else checks.FAILED))
            else:
                items.append(checks.Check(
                    "figures dependant des reseaux : regeneration", checks.NOT_RUN,
                    ["exige 3 reconstructions verifiees ; relancer avec --with-models"]))
    except (pipeline.ExportError, RuntimeError) as exc:
        items.append(checks.Check("regeneration", checks.FAILED, [str(exc)]))
    failed = report(items)
    print(f"verify : {'ECHEC' if failed else 'OK'} ({failed} controle(s) en echec)")
    return 1 if failed else 0


def cmd_export(args) -> int:
    with tempfile.TemporaryDirectory(prefix="c0_export_") as tmp:
        out = Layout(Path(tmp))
        try:
            info = pipeline.produce(out, REPO, with_models=args.with_models)
        except (pipeline.ExportError, RuntimeError) as exc:
            print(f"export refuse : {exc}")
            return 2
        drift = pipeline.compare(out, REPO, only=pipeline.source_paths())
        if not pipeline.is_clean(drift):
            report([diff_check("derive des tables sources : export refuse, depot inchange", drift)])
            return 2
        changed = pipeline.sync(out, REPO, exclude=pipeline.source_paths())
    print(f"export : {len(changed)} fichier(s) ecrit(s) ou modifie(s) ; modeles : {info['model_figures']}")
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
        print("reproduce refuse : --out doit etre hors du depot")
        return 2
    if out_dir.exists() and any(out_dir.iterdir()):
        print(f"reproduce refuse : {out_dir} n'est pas vide")
        return 2
    if args.full and args.no_models:
        print("reproduce refuse : --full et --no-models sont incompatibles")
        return 2
    out = Layout(out_dir)
    try:
        if args.full:
            info = pipeline.produce_full(out, REPO)
        else:
            info = pipeline.produce(out, REPO, with_models=not args.no_models)
    except (pipeline.ExportError, RuntimeError) as exc:
        print(f"reproduce refuse : {exc}")
        return 2
    if args.full:
        from src.experiments.runner_2d_nominal import NOMINAL_STORE
        stores = checks.Check.from_issues(
            f"{info['runs_compared']} runs reentraines : poids et metriques identiques au cache valide",
            pipeline.compare_stores(out.root / "store", NOMINAL_STORE))
        diff = diff_check(f"reproduction integrale ({out_dir}) : contenu scientifique identique au depot",
                          pipeline.compare_scientific(out, REPO))
        items = [stores, diff, *checks.run_static_checks(out)]
    else:
        diff = pipeline.compare(out, REPO)
        items = [diff_check(f"reproduction isolee ({out_dir}) identique au depot", diff),
                 *checks.run_static_checks(out)]
    failed = report(items)
    print(f"reproduce : {'ECHEC' if failed else 'OK'} ; modeles : {info['model_figures']}")
    return 1 if failed else 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("verify", "export"):
        p = sub.add_parser(name)
        p.add_argument("--with-models", action="store_true",
                       help="regenere aussi les figures qui exigent 3 reseaux reconstruits")
    p = sub.add_parser("reproduce")
    p.add_argument("--out", required=True, help="repertoire vide, hors du depot")
    p.add_argument("--no-models", action="store_true")
    p.add_argument("--full", action="store_true",
                   help="reproduction integrale : reentraine tout dans DIR/store (explicite, isole)")
    args = parser.parse_args(argv)
    return {"verify": cmd_verify, "export": cmd_export, "reproduce": cmd_reproduce}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
