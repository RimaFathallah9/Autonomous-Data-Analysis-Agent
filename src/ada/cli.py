"""`ada` command line: datasets | fetch | run | serve."""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="ada")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("datasets", help="list registered datasets and licenses")
    f = sub.add_parser("fetch", help="download + verify (or pin) a remote dataset")
    f.add_argument("name")
    r = sub.add_parser("run", help="run an experiment config")
    r.add_argument("config")
    r.add_argument("--out", default=None)
    s = sub.add_parser("serve", help="start the FastAPI server")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    a = p.parse_args(argv)

    if a.cmd == "datasets":
        from ada.datasets import load_registry

        for d in load_registry().values():
            print(f"{d.name:22} {d.kind:10} {d.license}")
    elif a.cmd == "fetch":
        from ada.datasets.registry import fetch_remote, load_registry

        print(fetch_remote(load_registry()[a.name]))
    elif a.cmd == "run":
        from ada.eval.runner import run_experiment

        run_experiment(a.config, a.out)
    elif a.cmd == "serve":
        import uvicorn

        uvicorn.run("ada.api:app", host=a.host, port=a.port)


if __name__ == "__main__":
    main()
