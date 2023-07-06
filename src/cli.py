from src.run import run
import click


@click.command()
@click.option("-e", "--exp-csv", required=True, type=click.Path(exists=True, resolve_path=True),
              help="File path of the csv file holding expression data.")
@click.option("-p", "--prior-csv", required=True, type=click.Path(exists=True, resolve_path=True),
              help="File path of the csv file to the prior matrix with entries in [1,2,3] for low, mid, hi values.")
@click.option("-o", "--out-dir", required=True, type=str, help="Output directory.")
@click.option("-t", "--num-iters", default=10000, type=int, help="Number of iterations.")
def infer(**kwargs):
    """Performing inference for ESQmodel."""
    run(**kwargs)


@click.group(name='esq')
def main():
    pass


main.add_command(infer)

if __name__ == '__main__':
    main()
