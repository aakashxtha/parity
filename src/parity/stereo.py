"""stereo.py — Symmetry canonicalization and geometry-based prochiral assignment.

This module operates on per-residue atom-position dicts (atomname -> np.ndarray)
and produces name remapping dicts.

Two public entry-points:
  canonicalize_symmetric_atoms()  — §9.2 ring-flip / resonance disambiguation
  assign_prochiral_atoms()        — §9.3 geometry-based CH2 hydrogen assignment
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


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class SymmetrySet:
    residues: list[str]            # canonical residue names this set applies to
    atom_pairs: list[list[str]]    # e.g. [["CD1","CD2"], ["CE1","CE2"]]
    coupled: bool                  # must swap as a unit
    reference: str                 # reference atom for deterministic ordering


@dataclass
class ProchiralDeclaration:
    residues: list[str]        # canonical residue names, or ["*"]
    center: str                # prochiral carbon
    dialect_names: list[str]   # [dialect_H_a, dialect_H_b]
    canonical_names: list[str] # [pdbv3_H_a, pdbv3_H_b]
    reference: list[str]       # [X, Y] — two distinct substituents


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_symmetry_sets() -> list[SymmetrySet]:
    """Load symmetry sets from data/symmetry.toml."""
    path = DATA_DIR / "symmetry.toml"
    with open(path, "rb") as fh:
        data = tomllib.load(fh)

    result: list[SymmetrySet] = []
    for entry in data.get("sets", []):
        result.append(SymmetrySet(
            residues=list(entry.get("residues", [])),
            atom_pairs=[list(pair) for pair in entry.get("atoms", [])],
            coupled=bool(entry.get("coupled", False)),
            reference=str(entry.get("reference", "")),
        ))
    return result


# ---------------------------------------------------------------------------
# Symmetric atom canonicalization  (§9.2)
# ---------------------------------------------------------------------------

def canonicalize_symmetric_atoms(
    resname: str,
    atom_positions: dict[str, "np.ndarray"],  # atomname -> xyz
    symmetry_sets: list[SymmetrySet],
) -> dict[str, str]:
    """Return {original_name -> canonical_name} for symmetric atom pairs.

    Only atoms that were *reassigned* are included; identity mappings are omitted.

    The rule (applied deterministically, regardless of input labeling):
      - For coupled pairs (PHE/TYR ring): compute the distance of the FIRST atom
        of each pair (CD1/CD2) to the reference (CG).  Assign CD1 to the nearer
        one.  If both pairs (CE1/CE2) need to flip with the first pair, flip them
        too.
      - For uncoupled pairs (ASP/GLU carboxylate, ARG guanidinium): for each pair
        independently, assign atom_pairs[i][0] (e.g. OD1) to the atom with the
        SMALLER distance to the reference.  Tie-break: smaller x coordinate; then
        y; then z.
    """
    import numpy as np  # lazy import

    mapping: dict[str, str] = {}

    for sym_set in symmetry_sets:
        if resname not in sym_set.residues:
            continue
        ref = sym_set.reference
        if ref not in atom_positions:
            continue  # can't canonicalize without reference

        ref_pos = atom_positions[ref]

        if sym_set.coupled:
            # All pairs must be flipped together.
            # Use the FIRST pair to determine orientation.
            first_pair = sym_set.atom_pairs[0]
            a_name, b_name = first_pair[0], first_pair[1]

            a_pos = atom_positions.get(a_name)
            b_pos = atom_positions.get(b_name)

            if a_pos is None or b_pos is None:
                continue  # insufficient data

            dist_a = float(np.linalg.norm(a_pos - ref_pos))
            dist_b = float(np.linalg.norm(b_pos - ref_pos))

            # Tie-break: lexicographic coordinate comparison
            def _lex_less(p1: "np.ndarray", p2: "np.ndarray") -> bool:
                for v1, v2 in zip(p1, p2):
                    if v1 < v2:
                        return True
                    if v1 > v2:
                        return False
                return False

            # Determine if we need to flip: we want pair[0]-canonical to be
            # the atom that is NEARER to reference.
            # If a is nearer (or equal and lex-less), a gets canonical pair[0] → no swap.
            # Otherwise swap.
            need_flip: bool
            if dist_a < dist_b:
                need_flip = False
            elif dist_b < dist_a:
                need_flip = True
            else:
                # tie-break by lex coord
                need_flip = _lex_less(b_pos, a_pos)

            if need_flip:
                for pair in sym_set.atom_pairs:
                    orig_0, orig_1 = pair[0], pair[1]
                    # Current: orig_0 is the atom that should be canonical[1]
                    # and orig_1 is the atom that should be canonical[0].
                    # We need to remap:
                    #   the actual atom at position orig_0 → should carry name orig_1
                    #   the actual atom at position orig_1 → should carry name orig_0
                    # In terms of what's in the file:
                    #   file has atom named orig_0 → rename to orig_1
                    #   file has atom named orig_1 → rename to orig_0
                    mapping[orig_0] = orig_1
                    mapping[orig_1] = orig_0
        else:
            # Uncoupled: handle each pair independently
            for pair in sym_set.atom_pairs:
                a_name, b_name = pair[0], pair[1]
                a_pos = atom_positions.get(a_name)
                b_pos = atom_positions.get(b_name)

                if a_pos is None or b_pos is None:
                    continue

                dist_a = float(np.linalg.norm(a_pos - ref_pos))
                dist_b = float(np.linalg.norm(b_pos - ref_pos))

                # Determine swap need: pair[0] canonical name goes to nearer atom
                need_flip: bool
                if dist_a < dist_b:
                    need_flip = False
                elif dist_b < dist_a:
                    need_flip = True
                else:
                    # tie-break: atom with smaller x (then y, then z) gets pair[0]
                    def _coord_lt(p1: "np.ndarray", p2: "np.ndarray") -> bool:
                        for v1, v2 in zip(p1, p2):
                            if v1 < v2:
                                return True
                            if v1 > v2:
                                return False
                        return False
                    need_flip = _coord_lt(b_pos, a_pos)

                if need_flip:
                    mapping[a_name] = b_name
                    mapping[b_name] = a_name

    return mapping


# ---------------------------------------------------------------------------
# Prochiral resolution  (§9.3)
# ---------------------------------------------------------------------------

def resolve_prochiral_pair(
    center_pos: "np.ndarray",  # position of the prochiral carbon C
    X_pos: "np.ndarray",       # position of distinct substituent X
    Y_pos: "np.ndarray",       # position of distinct substituent Y
    a_pos: "np.ndarray",       # position of atom a
    b_pos: "np.ndarray",       # position of atom b
) -> int:
    """Return 0 if 'a' gets canonical_names[0], 1 if 'b' gets canonical_names[0].

    Uses the scalar triple product from spec §9.3:
      s = (X - C) · [(Y - C) × (a - C)]

    Convention (FIXED — never change without a version bump):
      s > 0  → a gets canonical_names[0]  (e.g. HB2)
      s < 0  → b gets canonical_names[0]
      s == 0 → fallback: lexicographic coordinate comparison of a_pos, b_pos;
               the one with smaller x (then y, then z) gets canonical_names[0].

    The specific convention (positive s → first name) is arbitrary but FIXED.
    """
    import numpy as np  # lazy import

    XC = X_pos - center_pos
    YC = Y_pos - center_pos
    aC = a_pos - center_pos

    # Scalar triple product: XC · (YC × aC)
    cross = np.cross(YC, aC)
    s = float(np.dot(XC, cross))

    if s > 0.0:
        return 0  # a → canonical_names[0]
    elif s < 0.0:
        return 1  # b → canonical_names[0]
    else:
        # Degenerate: fall back to lexicographic coordinate order
        for va, vb in zip(a_pos, b_pos):
            if va < vb:
                return 0
            if va > vb:
                return 1
        return 0  # identical positions; assign a arbitrarily


def assign_prochiral_atoms(
    resname: str,
    dialect_name: str,
    atom_positions: dict[str, "np.ndarray"],  # dialect atom name -> xyz
    prochiral_declarations: list[ProchiralDeclaration],
) -> tuple[dict[str, str], list[str]]:
    """Return ({dialect_atomname -> canonical_atomname}, warnings) for prochiral pairs.

    For each prochiral declaration applicable to this residue:
    1. Look up both dialect atom names in atom_positions
    2. Look up center and reference atoms
    3. Call resolve_prochiral_pair to determine which gets which canonical name
    4. Build the mapping

    If coordinates are missing, fall back to positional/lexicographic assignment
    and add a warning string.
    """
    mapping: dict[str, str] = {}
    warnings: list[str] = []

    for decl in prochiral_declarations:
        # Check residue applicability
        if decl.residues != ["*"] and resname not in decl.residues:
            continue

        d_a, d_b = decl.dialect_names[0], decl.dialect_names[1]
        c_a, c_b = decl.canonical_names[0], decl.canonical_names[1]

        # Look up positions
        pos_a = atom_positions.get(d_a)
        pos_b = atom_positions.get(d_b)

        if pos_a is None and pos_b is None:
            # Neither hydrogen present — nothing to map
            continue

        if pos_a is None or pos_b is None:
            # Only one of the pair is present — assign by position in list
            if pos_a is not None:
                mapping[d_a] = c_a
            else:
                mapping[d_b] = c_b
            warnings.append(
                f"{resname} prochiral pair ({d_a},{d_b}): only one atom present, "
                f"fallback to positional assignment"
            )
            continue

        # Both present — attempt geometry-based assignment
        center_pos = atom_positions.get(decl.center)
        ref_x_pos = atom_positions.get(decl.reference[0]) if len(decl.reference) > 0 else None
        ref_y_pos = atom_positions.get(decl.reference[1]) if len(decl.reference) > 1 else None

        if center_pos is None or ref_x_pos is None or ref_y_pos is None:
            # Fallback: assign d_a -> c_a, d_b -> c_b (positional)
            mapping[d_a] = c_a
            mapping[d_b] = c_b
            warnings.append(
                f"{resname} prochiral pair ({d_a},{d_b}): missing center/reference "
                f"atoms, fallback to positional assignment"
            )
            continue

        idx = resolve_prochiral_pair(center_pos, ref_x_pos, ref_y_pos, pos_a, pos_b)
        if idx == 0:
            # a gets c_a, b gets c_b
            mapping[d_a] = c_a
            mapping[d_b] = c_b
        else:
            # b gets c_a, a gets c_b
            mapping[d_a] = c_b
            mapping[d_b] = c_a

    return mapping, warnings
