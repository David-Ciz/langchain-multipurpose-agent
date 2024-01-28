import logging

import click
import requests


@click.group()
def cli():
    pass



@cli.command()
def test_health():
    res = requests.get("http://127.0.0.1:8000/health")
    print(res.json())


@cli.command()
@click.argument("prompt")
def test_chat(prompt: str):
    res = requests.post("http://127.0.0.1:8000/chat", json={"text": prompt})
    print(res.json())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cli()
