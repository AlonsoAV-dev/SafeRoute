"""CLI de la optimización de buffer de 300 m con etiquetas y prueba fijas."""
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "search", "refine", "adapt", "finalize", "report"))
    args = parser.parse_args()
    if args.stage == "prepare":
        from buffer300.data import prepare
        prepare()
    else:
        from buffer300.modelos import run
        run(args.stage)
