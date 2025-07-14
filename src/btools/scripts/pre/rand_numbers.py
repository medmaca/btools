"""Example script to generate random numbers."""

import random

import click


def generate_random_numbers(count: int) -> None:
    """Generate a list of random numbers."""
    random_numbers = [random.randint(1, 100) for _ in range(count)]
    click.echo(f"Random Numbers: {random_numbers}")
