"""resolve.py — Parse a structure file and produce canonical atoms.

The main public function is resolve_structure(), which:
  1. Reads a PDB or mmCIF file with gemmi
  2. Detects (or applies overridden) dialect
  3. Maps every residue and atom to PDB v3 canonical identities
  4. Runs symmetry canonicalization and prochiral assignment
  5. Returns a CanonicalStructure

See PARITY_SPEC.md for the full spec; §9.1–9.4 are most relevant here.
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass, field
from collections import Counter
from typing import Optional

from parity.model import CanonicalAtom, CanonicalResidue
from parity import dialects as _dialects
from parity import stereo as _stereo


# ---------------------------------------------------------------------------
# Standard amino acids (PDB v3 names that need no mapping)
# ---------------------------------------------------------------------------

_STANDARD_AA: frozenset[str] = frozenset({
    "ALA", "ARG", "ASN", "ASP", "CYS",
    "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO",
    "SER", "THR", "TRP", "TYR", "VAL",
})


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class CanonicalStructure:
    """Result of resolving a structure file."""
    source_path: str
    dialect: str                    # detected dialect name
    dialect_confidence: str         # 'confident', 'ambiguous', 'insufficient evidence'
    dialect_alt: Optional[str]      # second-best dialect when ambiguous
    model_index: int                # which model was used (0-based)
    model_count: int                # total models in file

    # The main data: canonical atom list
    atoms: list[CanonicalAtom] = field(default_factory=list)

    # For reporting
    residues: list[CanonicalResidue] = field(default_factory=list)
    chains: list[str] = field(default_factory=list)

    # Issues found during resolution
    unknown_residues: list[dict] = field(default_factory=list)
    unknown_atoms: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Statistics collection (for dialect scoring)
# ---------------------------------------------------------------------------

def _collect_stats(model) -> dict:
    """Collect structure_stats dict for dialect scoring.

    Parameters
    ----------
    model : gemmi.Model
    """
    residue_names: Counter = Counter()
    atom_names: Counter = Counter()
    atom_in_residue: Counter = Counter()

    for chain in model:
        for residue in chain:
            rname = residue.name
            residue_names[rname] += 1
            for atom in residue:
                atom_names[atom.name] += 1
                atom_in_residue[(rname, atom.name)] += 1

    return {
        "residue_names": residue_names,
        "atom_names": atom_names,
        "atom_in_residue": atom_in_residue,
    }


# ---------------------------------------------------------------------------
# Main resolution function
# ---------------------------------------------------------------------------

def resolve_structure(
    path: str | pathlib.Path,
    *,
    dialect_override: Optional[str] = None,
    include_hydrogens: bool = False,
    model_n: int = 1,
) -> CanonicalStructure:
    """Parse a PDB or mmCIF file and return a CanonicalStructure.

    Parameters
    ----------
    path:
        Path to a .pdb or .cif file.
    dialect_override:
        Force a specific dialect name (e.g. 'charmm') instead of auto-detecting.
    include_hydrogens:
        When False (default), hydrogen/deuterium atoms are skipped.
    model_n:
        Which model to use (1-based).  If the file has multiple models and
        model_n == 1, a warning is added to the result.
    """
    import gemmi  # lazy import

    path = pathlib.Path(path)

    # 1. Parse structure
    structure = gemmi.read_structure(str(path))
    model_count = len(structure)
    model_index = model_n - 1  # convert to 0-based

    if model_index < 0 or model_index >= model_count:
        raise ValueError(
            f"model_n={model_n} out of range; file has {model_count} model(s)"
        )

    model = structure[model_index]

    warnings: list[str] = []
    if model_count > 1:
        warnings.append(
            f"File contains {model_count} models; using model {model_n} "
            f"(use model_n= to select a different one)"
        )

    # 2. Load dialects
    all_dialects = _dialects.load_all_dialects()

    # 3. Collect stats for dialect detection
    stats = _collect_stats(model)

    # 4. Detect or apply dialect override
    dialect_alt: Optional[str] = None
    if dialect_override is not None:
        if dialect_override not in all_dialects:
            raise ValueError(
                f"Unknown dialect '{dialect_override}'. "
                f"Available: {sorted(all_dialects.keys())}"
            )
        dialect_name = dialect_override
        dialect_confidence = "override"
    else:
        detection = _dialects.detect_dialect(all_dialects, stats)
        dialect_name = detection[0]
        dialect_confidence = detection[2]
        if len(detection) == 4:
            dialect_alt = detection[3]  # type: ignore[misc]

    table = all_dialects[dialect_name]

    # 5. Load symmetry sets and build prochiral declarations
    symmetry_sets = _stereo.load_symmetry_sets()
    prochiral_decls = _build_prochiral_decls(table)

    # 6. Walk structure and build canonical atoms
    atoms: list[CanonicalAtom] = []
    residues: list[CanonicalResidue] = []
    chains_seen: list[str] = []
    chain_ids_seen: set[str] = set()
    unknown_residues: list[dict] = []
    unknown_atoms: list[dict] = []

    for chain in model:
        chain_id = chain.name
        if chain_id not in chain_ids_seen:
            chains_seen.append(chain_id)
            chain_ids_seen.add(chain_id)

        for residue in chain:
            dialect_resname = residue.name
            seq_id = int(residue.seqid.num)
            icode_raw = str(residue.seqid.icode).strip()
            insertion_code = icode_raw if icode_raw and icode_raw != "\x00" else ""

            # --- Map residue name to canonical identity ---
            canonical_resname, protonation_state, modification = _resolve_residue(
                table, dialect_resname, unknown_residues,
                chain_id, seq_id, insertion_code
            )

            # Build canonical residue record
            can_res = CanonicalResidue(
                chain_id=chain_id,
                seq_id=seq_id,
                insertion_code=insertion_code,
                parent_name=canonical_resname,
                protonation_state=protonation_state,
                modification=modification,
                dialect_name=dialect_resname,
            )
            residues.append(can_res)

            # --- Collect atom positions for this residue (for stereo) ---
            # Keyed by DIALECT atom name (pre-mapping), per the spec note:
            # prochiral assignment uses dialect names to look up positions.
            residue_positions: dict[str, "np.ndarray"] = {}
            for atom in residue:
                if not include_hydrogens and atom.element.name in ("H", "D"):
                    continue
                import numpy as np
                residue_positions[atom.name] = np.array(
                    [atom.pos.x, atom.pos.y, atom.pos.z], dtype=float
                )

            # --- Prochiral assignment (dialect names -> canonical names) ---
            prochiral_map, prochiral_warns = _stereo.assign_prochiral_atoms(
                canonical_resname,
                dialect_name,
                residue_positions,
                prochiral_decls,
            )
            warnings.extend(prochiral_warns)

            # --- Symmetric atom canonicalization ---
            # Symmetry uses canonical atom names; we must apply it AFTER name mapping.
            # Build a canonical-name -> position dict first.
            # We do this in two passes:
            #   Pass 1: map dialect atom names to canonical names (excluding prochiral)
            #   Pass 2: apply symmetry canonicalization on those canonical names
            #   Pass 3: construct CanonicalAtom for each atom

            # Build per-atom canonical name before symmetry
            canonical_name_pre_sym: dict[str, str] = {}  # dialect_name -> canonical_name
            for atom in residue:
                if not include_hydrogens and atom.element.name in ("H", "D"):
                    continue
                d_aname = atom.name

                # Prochiral takes priority
                if d_aname in prochiral_map:
                    canonical_name_pre_sym[d_aname] = prochiral_map[d_aname]
                    continue

                # Global + per-residue atom mapping
                mapped = _dialects.get_atom_canonical(table, canonical_resname, d_aname)
                if mapped is not None:
                    canonical_name_pre_sym[d_aname] = mapped
                else:
                    # Handle star-notation nucleotide atoms (pdbv2 legacy)
                    if d_aname.endswith("*"):
                        canonical_name_pre_sym[d_aname] = d_aname[:-1] + "'"
                    else:
                        # No mapping: pass through as-is
                        canonical_name_pre_sym[d_aname] = d_aname

            # Build canonical-name -> position map for symmetry
            canon_pos: dict[str, "np.ndarray"] = {}
            for d_aname, c_aname in canonical_name_pre_sym.items():
                pos = residue_positions.get(d_aname)
                if pos is not None:
                    canon_pos[c_aname] = pos

            # Apply symmetry canonicalization
            sym_remap = _stereo.canonicalize_symmetric_atoms(
                canonical_resname, canon_pos, symmetry_sets
            )
            # sym_remap: {old_canonical_name -> new_canonical_name}

            # Pass 3: emit CanonicalAtom for each atom
            for atom in residue:
                if not include_hydrogens and atom.element.name in ("H", "D"):
                    continue

                d_aname = atom.name
                altloc_raw = atom.altloc
                altloc = str(altloc_raw) if altloc_raw and altloc_raw != "\x00" else ""

                # Final canonical atom name: pre-sym name, then apply sym remap
                pre_sym_name = canonical_name_pre_sym.get(d_aname, d_aname)
                final_name = sym_remap.get(pre_sym_name, pre_sym_name)

                element = atom.element.name

                # Track unknown atoms: those that had no mapping and are not in
                # standard sets; we flag them but still include them.
                if (
                    d_aname not in prochiral_map
                    and _dialects.get_atom_canonical(table, canonical_resname, d_aname) is None
                    and not d_aname.endswith("*")
                    and pre_sym_name == d_aname  # no mapping was found
                    and canonical_resname not in _STANDARD_AA
                ):
                    # Only flag if the residue itself was unknown
                    pass  # unknown atoms for known residues are expected (e.g. ligands)

                atoms.append(CanonicalAtom(
                    chain_id=chain_id,
                    seq_id=seq_id,
                    insertion_code=insertion_code,
                    canonical_resname=canonical_resname,
                    canonical_atomname=final_name,
                    altloc=altloc,
                    dialect_atomname=d_aname,
                    element=element,
                ))

    return CanonicalStructure(
        source_path=str(path),
        dialect=dialect_name,
        dialect_confidence=dialect_confidence,
        dialect_alt=dialect_alt,
        model_index=model_index,
        model_count=model_count,
        atoms=atoms,
        residues=residues,
        chains=chains_seen,
        unknown_residues=unknown_residues,
        unknown_atoms=unknown_atoms,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_residue(
    table: _dialects.DialectTable,
    dialect_resname: str,
    unknown_residues: list[dict],
    chain_id: str,
    seq_id: int,
    insertion_code: str,
) -> tuple[str, str, str]:
    """Map a dialect residue name to (canonical_resname, protonation, modification).

    Returns the triple.  For unknown residues, returns a pass-through and records
    the residue in unknown_residues.
    """
    # Check dialect table
    result = _dialects.get_residue_canonical(table, dialect_resname)
    if result is not None:
        return result

    # Standard AAs are already canonical — no table entry needed
    if dialect_resname in _STANDARD_AA:
        return (dialect_resname, "", "")

    # Unknown: pass-through with empty protonation/modification
    unknown_residues.append({
        "chain": chain_id,
        "seq_id": seq_id,
        "insertion_code": insertion_code,
        "name": dialect_resname,
    })
    return (dialect_resname, "", "")


def _build_prochiral_decls(
    table: _dialects.DialectTable,
) -> list[_stereo.ProchiralDeclaration]:
    """Convert raw TOML prochiral dicts from a dialect table into ProchiralDeclaration objects."""
    result: list[_stereo.ProchiralDeclaration] = []
    for raw in table.prochiral:
        result.append(_stereo.ProchiralDeclaration(
            residues=list(raw.get("residues", [])),
            center=str(raw.get("center", "")),
            dialect_names=list(raw.get("dialect", [])),
            canonical_names=list(raw.get("canonical", [])),
            reference=list(raw.get("reference", [])),
        ))
    return result
