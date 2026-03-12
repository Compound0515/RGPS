import argparse

from rgps.cli import (
    run_gen,
    run_opt,
    run_perturb,
    run_select,
    run_submit,
)


def main():
    parser = argparse.ArgumentParser(description="RGPS Workflow Tool")
    subparsers = parser.add_subparsers(
        dest="command",
        help="Available commands",
        required=True,
    )

    # Register subcommands
    run_gen.register(subparsers)
    run_perturb.register(subparsers)
    run_opt.register(subparsers)
    run_select.register(subparsers)
    run_submit.register(subparsers)

    args = parser.parse_args()

    # Dispatch
    if args.command == "gen":
        run_gen.execute(args)
    elif args.command == "perturb":
        run_perturb.execute(args)
    elif args.command == "opt":
        run_opt.execute(args)
    elif args.command == "select":
        run_select.execute(args)
    elif args.command == "submit":
        run_submit.execute(args)


if __name__ == "__main__":
    main()
