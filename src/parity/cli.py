"""parity CLI — entry point.

Heavy imports (gemmi, numpy) are done lazily inside command functions so that
`parity --help` stays fast (< 200 ms).
"""
from __future__ import annotations

from typing import Optional

import typer
from typing import Annotated

app = typer.Typer(
    name="parity",
    help="Fast, read-only tool for auditing and comparing PDB/mmCIF structure files.",
    no_args_is_help=True,
    add_completion=True,
    invoke_without_command=True,
)


def _version_callback(value: bool) -> None:
    if value:
        from parity import __version__

        typer.echo(f"parity {__version__}")
        raise typer.Exit()


@app.callback()
def _main(
    ctx: typer.Context,
    version: Optional[bool] = typer.Option(  # noqa: B008
        None,
        "--version",
        callback=_version_callback,
        is_eager=True,
        expose_value=False,
        help="Show version and exit.",
    ),
) -> None:
    """parity — audit and compare PDB/mmCIF structure files."""


# ---------------------------------------------------------------------------
# Shared options
# ---------------------------------------------------------------------------

_JsonOpt = Annotated[
    bool,
    typer.Option("--json", help="Emit machine-readable JSON instead of human text."),
]
_HydrogensOpt = Annotated[
    bool,
    typer.Option("--hydrogens", help="Include hydrogen atoms in comparisons."),
]
_DialectOpt = Annotated[
    Optional[str],
    typer.Option(
        "--dialect",
        help="Force a specific naming dialect (pdbv3, pdbv2, amber, charmm). "
        "Auto-detected when omitted.",
        metavar="DIALECT",
    ),
]
_ModelOpt = Annotated[
    int,
    typer.Option(
        "--model",
        help="Which MODEL record (1-based) to use from multi-model files.",
        metavar="N",
    ),
]
_QuietOpt = Annotated[
    bool,
    typer.Option("--quiet", "-q", help="Suppress informational output; only exit code."),
]
_NoColorOpt = Annotated[
    bool,
    typer.Option("--no-color", help="Disable ANSI colour in output."),
]


# ---------------------------------------------------------------------------
# parity check
# ---------------------------------------------------------------------------


@app.command()
def check(
    file: Annotated[
        str,
        typer.Argument(help="Structure file to audit (PDB or mmCIF)."),
    ],
    json: _JsonOpt = False,
    hydrogens: _HydrogensOpt = False,
    dialect: _DialectOpt = None,
    model: _ModelOpt = 1,
    quiet: _QuietOpt = False,
    no_color: _NoColorOpt = False,
) -> None:
    """Audit a single structure file for internal consistency."""
    try:
        from parity.check import check_structure
        from parity.render.json_out import render_check_json
        from parity.render.human import render_check
        from rich.console import Console

        report = check_structure(
            file,
            dialect_override=dialect,
            include_hydrogens=hydrogens,
            model_n=model,
        )

        if json:
            typer.echo(render_check_json(report))
        else:
            console = Console(no_color=no_color)
            render_check(report, console, quiet=quiet)

        raise typer.Exit(code=1 if report.issue_count > 0 else 0)

    except typer.Exit:
        raise
    except Exception as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=2) from exc


# ---------------------------------------------------------------------------
# parity diff
# ---------------------------------------------------------------------------


@app.command()
def diff(
    file_a: Annotated[
        str,
        typer.Argument(help="First structure file (PDB or mmCIF)."),
    ],
    file_b: Annotated[
        str,
        typer.Argument(help="Second structure file to compare against FILE_A."),
    ],
    # diff-specific options
    level: Annotated[
        str,
        typer.Option(
            "--level",
            help="Comparison level: naming | topology | chemistry | geometry | all.",
            metavar="LEVEL",
        ),
    ] = "all",
    rmsd_threshold: Annotated[
        float,
        typer.Option(
            "--rmsd-threshold",
            help="RMSD threshold (Angstroms) below which coordinate differences are ignored.",
            metavar="FLOAT",
        ),
    ] = 0.0,
    ignore_water: Annotated[
        bool,
        typer.Option("--ignore-water", help="Exclude water molecules from comparison."),
    ] = False,
    ignore_hetatm: Annotated[
        bool,
        typer.Option("--ignore-hetatm", help="Exclude HETATM records from comparison."),
    ] = False,
    # shared options
    json: _JsonOpt = False,
    hydrogens: _HydrogensOpt = False,
    dialect: _DialectOpt = None,
    model: _ModelOpt = 1,
    quiet: _QuietOpt = False,
    no_color: _NoColorOpt = False,
) -> None:
    """Compare two structure files and report differences."""
    try:
        from parity.resolve import resolve_structure
        from parity.diff import diff_structures
        from parity.render.json_out import render_diff_json
        from parity.render.human import render_diff
        from rich.console import Console

        cs_a = resolve_structure(
            file_a,
            dialect_override=dialect,
            include_hydrogens=hydrogens,
            model_n=model,
        )
        cs_b = resolve_structure(
            file_b,
            dialect_override=dialect,
            include_hydrogens=hydrogens,
            model_n=model,
        )

        report = diff_structures(
            cs_a, cs_b,
            level=level,
            rmsd_threshold=rmsd_threshold,
            ignore_water=ignore_water,
            ignore_hetatm=ignore_hetatm,
        )

        if json:
            typer.echo(render_diff_json(report))
        else:
            console = Console(no_color=no_color)
            render_diff(report, console, quiet=quiet)

        raise typer.Exit(code=0 if report.equivalent else 1)

    except typer.Exit:
        raise
    except Exception as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=2) from exc
