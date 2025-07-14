"""CLI entry points for the btools package."""

import click

from btools.cli.cli_commands import pre_group

# see https://click.palletsprojects.com/en/stable/commands-and-groups/#group-invocation-without-command


@click.group(context_settings={"help_option_names": ["-h", "--help"]}, invoke_without_command=True)
@click.version_option(None, "-V", "--version", message="%(prog)s version %(version)s")
@click.pass_context
def cli(ctx: click.Context):
    """CLI example for the template_cli package."""
    if ctx.invoked_subcommand is None:
        click.echo("Welcome to this example CLI Tool!")
        click.echo(ctx.get_help())
        return


# Register commands
cli.add_command(pre_group, name="pre")
