import click
from utils.config import load_config, output_dir, log_setup
from models import yolo, rfdetr, retinanet
from trackers import bytetrack, norfair


@click.group()
def cli():
    pass

@cli.command()
@click.option("--detector", "-d", type=click.Choice(["yolo", "rfdetr", "retinanet"]), default="yolo", help="Detector to use for demo video")
@click.option("--tracker", "-t", type=click.Choice(["bytetrack", "norfair"]), default="bytetrack", help="Tracker to use for demo video")
@click.option("--device", "-D", type=click.Choice(["cpu", "cuda", "mps"]), help="Override device")
def demo(detector: str, tracker: str, device: str):
    config = load_config()
    output_dir(config)
    log_setup(config["logging"]["verbosity"])
    