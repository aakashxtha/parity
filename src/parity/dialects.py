"""dialects.py — Load and query force-field dialect TOML tables.

Each dialect file lives under data/dialects/ and maps FF-specific residue/atom
names to the canonical PDB v3 form.  This module provides:

  - DialectTable dataclass holding parsed dialect data
  - load_all_dialects()   -> dict[name, DialectTable]
  - get_residue_canonical() / get_atom_canonical() for point lookups
  - score_dialect() / detect_dialect() for automatic dialect detection
"""
from __future__ import annotations

import sys
import pathlib
from dataclasses import dataclass, field

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

DATA_DIR = pathlib.Path(__file__).parent / "data"
_DIALECTS_DIR = DATA_DIR / "dialects"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class DialectTable:
    name: str
    description: str
    reference: str
    # residue name -> (canonical_name, protonation_state, modification)
    residues: dict[str, tuple[str, str, str]]
    # atom mappings: global dict + per-residue dicts
    global_atoms: dict[str, str]
    per_residue_atoms: dict[str, dict[str, str]]  # canonical_resname -> {dialect_atom -> canonical_atom}
    # prochiral declarations (list of dicts from TOML)
    prochiral: list[dict]
    # markers for detection scoring
    markers: list[dict]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _parse_dialect(path: pathlib.Path) -> DialectTable:
    """Parse a single dialect TOML file into a DialectTable."""
    with open(path, "rb") as fh:
        data = tomllib.load(fh)

    meta = data.get("meta", {})

    # --- residue mappings ---
    residues: dict[str, tuple[str, str, str]] = {}
    raw_residues = data.get("residues", {})
    for dialect_name, info in raw_residues.items():
        if isinstance(info, dict):
            residues[dialect_name] = (
                info.get("canonical", dialect_name),
                info.get("protonation", ""),
                info.get("modification", ""),
            )
        # lists (e.g. standard_amino_acids) are skipped

    # --- atom mappings ---
    raw_atoms = data.get("atoms", {})
    global_atoms: dict[str, str] = {}
    per_residue_atoms: dict[str, dict[str, str]] = {}

    for key, val in raw_atoms.items():
        if key == "global":
            if isinstance(val, dict):
                global_atoms = dict(val)
        else:
            # key is a canonical residue name
            if isinstance(val, dict):
                per_residue_atoms[key] = dict(val)

    # --- prochiral ---
    prochiral: list[dict] = data.get("prochiral", [])

    # --- markers ---
    markers: list[dict] = data.get("markers", [])

    return DialectTable(
        name=meta.get("name", path.stem),
        description=meta.get("description", ""),
        reference=meta.get("reference", ""),
        residues=residues,
        global_atoms=global_atoms,
        per_residue_atoms=per_residue_atoms,
        prochiral=prochiral,
        markers=markers,
    )


def load_all_dialects() -> dict[str, DialectTable]:
    """Load all dialect TOML files from data/dialects/ directory.

    Returns a dict keyed by dialect name (from [meta] name field).
    """
    result: dict[str, DialectTable] = {}
    for toml_path in sorted(_DIALECTS_DIR.glob("*.toml")):
        table = _parse_dialect(toml_path)
        result[table.name] = table
    return result


# ---------------------------------------------------------------------------
# Point lookups
# ---------------------------------------------------------------------------

def get_residue_canonical(
    table: DialectTable, dialect_resname: str
) -> tuple[str, str, str] | None:
    """Return (canonical_name, protonation, modification) or None if unknown."""
    return table.residues.get(dialect_resname)


def get_atom_canonical(
    table: DialectTable, canonical_resname: str, dialect_atomname: str
) -> str | None:
    """Return canonical PDB v3 atom name, or None if no mapping exists.

    Per-residue mapping takes precedence over global mapping.
    """
    # Per-residue first
    per_res = table.per_residue_atoms.get(canonical_resname, {})
    if dialect_atomname in per_res:
        return per_res[dialect_atomname]
    # Then global
    if dialect_atomname in table.global_atoms:
        return table.global_atoms[dialect_atomname]
    return None


# ---------------------------------------------------------------------------
# Dialect detection
# ---------------------------------------------------------------------------

def score_dialect(table: DialectTable, structure_stats: dict) -> float:
    """Score how well a structure matches a dialect's markers.

    structure_stats keys:
      'residue_names': Counter of residue names seen
      'atom_names': Counter of atom names seen
      'atom_in_residue': Counter of (residue_name, atom_name) tuples

    Returns normalized score (0.0 to 1.0).
    Normalization: weight of matching markers / total possible weight.
    Markers with zero opportunities are excluded from the denominator.
    """
    residue_names = structure_stats.get("residue_names", {})
    atom_names = structure_stats.get("atom_names", {})
    atom_in_residue = structure_stats.get("atom_in_residue", {})

    matched_weight = 0.0
    total_weight = 0.0

    for marker in table.markers:
        kind = marker.get("kind", "")
        value = marker.get("value", "")
        weight = float(marker.get("weight", 1))

        if kind == "residue_name":
            if residue_names.get(value, 0) > 0:
                # Opportunity exists (name present) → include in denominator
                total_weight += weight
                matched_weight += weight
            # No opportunity (name absent) → skip entirely; don't penalize
        elif kind == "atom_name":
            if atom_names.get(value, 0) > 0:
                total_weight += weight
                matched_weight += weight
            # No occurrence → no opportunity → skip
        elif kind == "atom_in_residue":
            residue = marker.get("residue", "")
            pair = (residue, value)
            if atom_in_residue.get(pair, 0) > 0:
                total_weight += weight
                matched_weight += weight
            else:
                # Check if the residue appears at all (opportunity exists)
                residue_present = residue_names.get(residue, 0) > 0
                if residue_present:
                    # Residue is present but the atom is not → miss, add to denom
                    total_weight += weight
                # Residue absent → zero opportunity → skip

    if total_weight == 0.0:
        return 0.0
    return matched_weight / total_weight


def detect_dialect(
    dialects: dict[str, DialectTable],
    structure_stats: dict,
) -> tuple[str, float, str] | tuple[str, float, str, str]:
    """Identify the most likely dialect.

    Returns (dialect_name, top_score, confidence_label) or
            (dialect_name, top_score, confidence_label, second_dialect_name)
    when confidence_label is 'ambiguous'.

    confidence_label is one of:
      'confident'           — top score beats second by >= 0.3
      'ambiguous'           — top score is close to second
      'insufficient evidence' — no markers matched at all
    """
    scores: dict[str, float] = {
        name: score_dialect(table, structure_stats)
        for name, table in dialects.items()
    }

    # Sort by descending score
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

    if not ranked:
        return ("unknown", 0.0, "insufficient evidence")

    top_name, top_score = ranked[0]

    # If no markers matched anything
    if top_score == 0.0:
        return (top_name, top_score, "insufficient evidence")

    second_name = ranked[1][0] if len(ranked) > 1 else None
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0

    gap = top_score - second_score

    if gap >= 0.3:
        return (top_name, top_score, "confident")
    else:
        if second_name is not None:
            return (top_name, top_score, "ambiguous", second_name)
        return (top_name, top_score, "ambiguous")
