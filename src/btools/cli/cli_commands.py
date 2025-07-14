"""Defines all CLI commands for the template-cli application."""

import click

from btools.scripts.pre.rand_numbers import generate_random_numbers


# Random number generation command
@click.command("random_num")
@click.option("--count", "-c", default=10, help="Number of random numbers to generate.")
def rand_gen(count: int):
    """Generate a list of random numbers using the generate_random_numbers function."""
    generate_random_numbers(count)
    generate_random_numbers(count)
