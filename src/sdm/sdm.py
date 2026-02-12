import click

from sdm.cli.ablation_study import ablation_study
from sdm.cli.beir import beir
from sdm.cli.core import core
from sdm.cli.core_exact import core_exact


@click.group()
def main():
    """
    Score Distribution Model (SDM)

    Predicting ranking quality in large-scale retrieval using score distributions modeling.
    """
    pass


main.add_command(core)
main.add_command(beir)
main.add_command(core_exact)
main.add_command(ablation_study)


if __name__ == "__main__":
    main()
