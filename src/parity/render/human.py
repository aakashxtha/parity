"""human.py — Rich-based human-readable output renderer.

Implements the spec §7.2 format for `parity check` and §8.1 for `parity diff`.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from rich.console import Console
from rich.text import Text

if TYPE_CHECKING:
    from parity.check import CheckReport
    from parity.diff import DiffReport


def render_check(report: "CheckReport", console: Console, quiet: bool = False) -> None:
    """Render a CheckReport to the console in spec §7.2 format.

    Example output::

      dialect     CHARMM  (confident)
                    HSD/HSE/HSP · HB1/HB2 · OT1/OT2 termini · TIP3 water

      contents    2 chains · 487 residues · 7,812 atoms
                  1 ligand (LIG) · 2 Zn · 41 waters

      protonation HIS: 9 HSD, 4 HSE, 1 HSP
                  CYS: 6 in 3 disulfides, 4 free

      ! gap        chain A 87-93 missing (numbering + CA-CA distance agree)
      ! altloc     3 residues carry alternate locations

      x 3 issues
    """
    if quiet:
        return

    LABEL_WIDTH = 14

    def _continuation() -> str:
        return " " * LABEL_WIDTH

    # -------------------------------------------------------------------
    # Dialect line
    # -------------------------------------------------------------------
    dialect_line = Text()
    dialect_line.append(f"  {'dialect':<{LABEL_WIDTH - 2}}")
    dialect_line.append(report.dialect.upper(), style="bold")
    dialect_line.append(f"  ({report.dialect_confidence})")
    console.print(dialect_line)

    # Evidence line (e.g. "HSD/HSE/HSP · HB1/HB2 · TIP3 water")
    if report.dialect_evidence:
        ev_str = " · ".join(report.dialect_evidence)
        console.print(f"{_continuation()}{ev_str}", style="dim")

    # Mixed-dialect warning
    if report.mixed_dialect:
        console.print(f"{_continuation()}[yellow]! chains use different dialects[/yellow]")

    console.print()

    # -------------------------------------------------------------------
    # Contents line
    # -------------------------------------------------------------------
    chain_word = "chain" if report.chain_count == 1 else "chains"
    res_word = "residue" if report.residue_count == 1 else "residues"
    contents_line = Text()
    contents_line.append(f"  {'contents':<{LABEL_WIDTH - 2}}")
    contents_line.append(
        f"{report.chain_count} {chain_word} · "
        f"{report.residue_count:,} {res_word} · "
        f"{report.atom_count:,} atoms"
    )
    console.print(contents_line)

    # Ligands / ions / waters sub-line
    extras: list[str] = []
    if report.ligand_names:
        n = len(report.ligand_names)
        lig_word = "ligand" if n == 1 else "ligands"
        names_str = ", ".join(report.ligand_names)
        extras.append(f"{n} {lig_word} ({names_str})")
    if report.ion_names:
        for ion in report.ion_names:
            extras.append(ion)
    if report.water_count:
        extras.append(f"{report.water_count:,} waters")

    if extras:
        console.print(f"{_continuation()}{' · '.join(extras)}")

    console.print()

    # -------------------------------------------------------------------
    # Protonation
    # -------------------------------------------------------------------
    his = report.his_counts
    total_his = sum(his.values())
    cys_total = report.cys_disulfide + report.cys_free

    if total_his > 0 or cys_total > 0 or report.has_nonstandard_protonation:
        label_prefix = f"  {'protonation':<{LABEL_WIDTH - 2}}"

        if total_his > 0:
            parts: list[str] = []
            if his.get('delta', 0):
                parts.append(f"{his['delta']} HSD")
            if his.get('epsilon', 0):
                parts.append(f"{his['epsilon']} HSE")
            if his.get('both', 0):
                parts.append(f"{his['both']} HSP")
            console.print(f"{label_prefix}HIS: {', '.join(parts)}")
        else:
            console.print(label_prefix)

        if cys_total > 0:
            cys_parts: list[str] = []
            if report.cys_disulfide > 0:
                ss_bonds = report.cys_disulfide // 2
                cys_parts.append(
                    f"{report.cys_disulfide} in {ss_bonds} disulfide{'s' if ss_bonds != 1 else ''}"
                )
            if report.cys_free > 0:
                cys_parts.append(f"{report.cys_free} free")
            console.print(f"{_continuation()}CYS: {', '.join(cys_parts)}")

        if report.has_nonstandard_protonation:
            console.print(f"{_continuation()}non-standard protonation states present")

        console.print()

    # -------------------------------------------------------------------
    # Issues
    # -------------------------------------------------------------------
    has_issues = report.issue_count > 0

    for cb in report.chain_breaks:
        reason = cb['reason']
        if reason == 'both':
            reason_str = 'numbering + CA-CA distance agree'
        elif reason == 'numbering':
            reason_str = 'numbering gap'
        else:
            reason_str = 'CA-CA distance gap'
        issue_line = Text()
        issue_line.append("  ", style="")
        issue_line.append("!", style="yellow bold")
        issue_line.append(f" {'gap':<{LABEL_WIDTH - 4}}", style="")
        issue_line.append(
            f"chain {cb['chain']} {cb['from_seq']}-{cb['to_seq']} "
            f"missing ({reason_str})"
        )
        console.print(issue_line)

    if report.altloc_residues:
        issue_line = Text()
        issue_line.append("  ")
        issue_line.append("!", style="yellow bold")
        n = len(report.altloc_residues)
        issue_line.append(f" {'altloc':<{LABEL_WIDTH - 4}}")
        issue_line.append(f"{n} residue{'s' if n != 1 else ''} carry alternate locations")
        console.print(issue_line)

    for unk in report.unknown_residues:
        issue_line = Text()
        issue_line.append("  ")
        issue_line.append("!", style="yellow bold")
        issue_line.append(f" {'unknown':<{LABEL_WIDTH - 4}}")
        issue_line.append(
            f"{unk['resname']} (chain {unk['chain']}, seq {unk['seq_id']}) "
            f"matches no known dialect"
        )
        console.print(issue_line)

    for unk in report.unknown_atoms:
        issue_line = Text()
        issue_line.append("  ")
        issue_line.append("!", style="yellow bold")
        issue_line.append(f" {'unknown':<{LABEL_WIDTH - 4}}")
        atom_info = f"{unk.get('resname', '?')}:{unk.get('atom', '?')}"
        issue_line.append(f"{atom_info} matches no known dialect")
        console.print(issue_line)

    if report.missing_atoms:
        issue_line = Text()
        issue_line.append("  ")
        issue_line.append("!", style="yellow bold")
        n = len(report.missing_atoms)
        issue_line.append(f" {'missing':<{LABEL_WIDTH - 4}}")
        issue_line.append(f"{n} residue{'s' if n != 1 else ''} have missing heavy atoms")
        console.print(issue_line)

    # Warnings (non-issue advisories)
    for w in report.warnings:
        warn_line = Text()
        warn_line.append("  ")
        warn_line.append("~", style="yellow")
        warn_line.append(f" {'warning':<{LABEL_WIDTH - 4}}")
        warn_line.append(w)
        console.print(warn_line)

    console.print()

    # -------------------------------------------------------------------
    # Summary line
    # -------------------------------------------------------------------
    summary = Text()
    summary.append("  ")
    if has_issues:
        summary.append("✗", style="red bold")
        summary.append(f" {report.issue_count} issue{'s' if report.issue_count != 1 else ''}", style="red")
    else:
        summary.append("✓", style="green bold")
        summary.append(" OK", style="green")
    console.print(summary)


def render_diff(report: "DiffReport", console: Console, quiet: bool = False) -> None:
    """Render a DiffReport to the console in spec §8.1 format."""
    if quiet:
        return

    from collections import defaultdict
    from parity.model import DiffLevel

    LABEL_WIDTH = 14

    def _lbl(s: str) -> str:
        return f"  {s:<{LABEL_WIDTH - 2}}"

    # --- Chemistry ---
    chem_line = Text()
    chem_line.append(_lbl("chemistry"))
    if report.chemistry_count == 0:
        chem_line.append("identical", style="green")
    else:
        chem_line.append(f"{report.chemistry_count} difference{'s' if report.chemistry_count != 1 else ''}", style="red")
    console.print(chem_line)

    # --- Topology ---
    topo_line = Text()
    topo_line.append(_lbl("topology"))
    if report.topology_count == 0:
        topo_line.append("identical", style="green")
    else:
        topo_line.append(f"{report.topology_count} difference{'s' if report.topology_count != 1 else ''}", style="red")
    console.print(topo_line)

    # --- Geometry ---
    geo_line = Text()
    geo_line.append(_lbl("geometry"))
    if report.geometry_rmsd is None:
        geo_line.append("not computed")
    else:
        geo_line.append(f"{report.geometry_rmsd:.3f} Å heavy-atom RMSD")
    console.print(geo_line)

    # --- Naming summary ---
    naming_diffs = [d for d in report.differences if d.level == DiffLevel.NAMING]
    if naming_diffs:
        nam_line = Text()
        nam_line.append(_lbl("naming"))
        nam_line.append(
            f"{report.naming_atom_count} atoms across {report.naming_residue_count} residues"
            if report.naming_atom_count > 0
            else f"{len(naming_diffs)} name differences"
        )
        console.print(nam_line)

    console.print()

    # --- Individual difference lines ---
    # Group naming atom diffs by (from_val, to_val) for summary
    naming_atom_groups: dict = defaultdict(int)
    for d in report.differences:
        if d.level == DiffLevel.NAMING and d.kind == 'atom_name':
            key = (d.from_val or '', d.to_val or '')
            naming_atom_groups[key] += 1

    for d in report.differences:
        if d.level == DiffLevel.NAMING and d.kind == 'atom_name':
            continue  # printed in groups below
        line = Text()
        if d.level == DiffLevel.CHEMISTRY:
            line.append("  ~ ", style="yellow bold")
        elif d.level == DiffLevel.TOPOLOGY:
            if d.kind in ('residue_added', 'chain_added', 'atom_added'):
                line.append("  + ", style="green bold")
            else:
                line.append("  - ", style="red bold")
        elif d.level == DiffLevel.GEOMETRY:
            line.append("  ~ ", style="dim")
        else:
            line.append("    ")
        line.append(d.message)
        console.print(line)

    for (from_v, to_v), cnt in sorted(naming_atom_groups.items()):
        line = Text()
        line.append("    ")
        line.append(f"{from_v} -> {to_v}", style="dim")
        line.append(f"   {cnt} atom{'s' if cnt != 1 else ''}")
        console.print(line)

    console.print()

    # --- Final equivalence line ---
    summary = Text()
    summary.append("  ")
    if report.equivalent:
        summary.append("=", style="green bold")
        summary.append(" chemically equivalent", style="green")
    else:
        summary.append("✗", style="red bold")
        summary.append(" structures are not equivalent", style="red")
    console.print(summary)
