import argparse

import satstash

from satstash.tui import SatStashApp


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="satstash")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--version", action="store_true")
    args = parser.parse_args(argv)

    if args.version:
        print(f"satstash {satstash.__version__} ({satstash.__file__})")
        return 0

    app = SatStashApp(debug=args.debug)
    app.run()
    return 0
