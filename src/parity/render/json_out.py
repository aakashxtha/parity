"""json_out.py — JSON output renderer for parity commands.

Implements spec §11.5 serialisation format.
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from parity.check import CheckReport
    from parity.diff import DiffReport


def render_check_json(report: "CheckReport") -> str:
    """Serialize a CheckReport to a JSON string (spec §11.5 format).

    Returns a pretty-printed JSON string.
    """
    his = report.his_counts
    payload: dict = {
        "parity_version": report.parity_version,
        "schema_version": 1,
        "command": "check",
        "file": report.source_path,
        "dialect": report.dialect,
        "dialect_confidence": report.dialect_confidence,
        "dialect_alt": report.dialect_alt,
        "dialect_evidence": report.dialect_evidence,
        "mixed_dialect": report.mixed_dialect,
        "chain_dialects": {
            chain_id: {"dialect": d, "confidence": c}
            for chain_id, (d, c) in report.chain_dialects.items()
        },
        "equivalent": None,
        "summary": {
            "chains": report.chain_count,
            "residues": report.residue_count,
            "atoms": report.atom_count,
            "ligands": report.ligand_names,
            "ions": report.ion_names,
            "waters": report.water_count,
        },
        "protonation": {
            "his": {
                "delta": his.get("delta", 0),
                "epsilon": his.get("epsilon", 0),
                "both": his.get("both", 0),
            },
            "cys": {
                "disulfide": report.cys_disulfide,
                "free": report.cys_free,
            },
            "has_nonstandard": report.has_nonstandard_protonation,
        },
        "issues": _serialize_issues(report),
        "warnings": report.warnings,
        "issue_count": report.issue_count,
        "exit_code": 1 if report.issue_count > 0 else 0,
    }
    return json.dumps(payload, indent=2)


def render_diff_json(report: "DiffReport") -> str:
    """Serialize a DiffReport to a JSON string (spec §11.5 format)."""
    from parity.model import DiffLevel

    differences = [
        {
            "level": d.level.value,
            "kind": d.kind,
            "chain": d.chain,
            "seq_id": d.seq_id,
            "insertion_code": d.insertion_code,
            "from": d.from_val,
            "to": d.to_val,
            "atom_name": d.atom_name,
            "count": d.count,
            "rmsd": d.rmsd,
            "message": d.message,
        }
        for d in report.differences
    ]

    payload: dict = {
        "parity_version": report.parity_version,
        "schema_version": 1,
        "command": "diff",
        "files": [report.file_a, report.file_b],
        "dialect_a": report.dialect_a,
        "dialect_b": report.dialect_b,
        "equivalent": report.equivalent,
        "summary": {
            "chemistry": report.chemistry_count,
            "topology": report.topology_count,
            "geometry": {"heavy_atom_rmsd": report.geometry_rmsd},
            "naming": {
                "atoms": report.naming_atom_count,
                "residues": report.naming_residue_count,
            },
        },
        "differences": differences,
        "warnings": report.warnings,
        "exit_code": 0 if report.equivalent else 1,
    }
    return json.dumps(payload, indent=2)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _serialize_issues(report: "CheckReport") -> list[dict]:
    """Flatten all issue types into a single list of dicts."""
    issues: list[dict] = []

    for cb in report.chain_breaks:
        issues.append({
            "type": "chain_break",
            "chain": cb["chain"],
            "from_seq": cb["from_seq"],
            "to_seq": cb["to_seq"],
            "reason": cb["reason"],
        })

    for ar in report.altloc_residues:
        issues.append({
            "type": "altloc",
            "chain": ar["chain"],
            "seq_id": ar["seq_id"],
            "resname": ar["resname"],
        })

    for unk in report.unknown_residues:
        issues.append({
            "type": "unknown_residue",
            "chain": unk.get("chain", ""),
            "seq_id": unk.get("seq_id", 0),
            "resname": unk.get("resname", ""),
        })

    for unk in report.unknown_atoms:
        issues.append({
            "type": "unknown_atom",
            "chain": unk.get("chain", ""),
            "seq_id": unk.get("seq_id", 0),
            "resname": unk.get("resname", ""),
            "atom": unk.get("atom", ""),
        })

    for ma in report.missing_atoms:
        issues.append({
            "type": "missing_atoms",
            "chain": ma["chain"],
            "seq_id": ma["seq_id"],
            "resname": ma["resname"],
            "missing": ma["missing"],
        })

    return issues
