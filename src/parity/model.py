"""model.py — core data model for parity.

All canonical identities are expressed in PDB v3 naming.  Every other dialect
maps *to* these forms; no dialect maps directly to any other dialect.

See PARITY_SPEC.md §9.1 for the canonical-key specification.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


# ---------------------------------------------------------------------------
# Difference level taxonomy (spec §8.2)
# ---------------------------------------------------------------------------

class DiffLevel(str, Enum):
    """Classification of a detected difference between two structures.

    Levels are ordered from least to most severe.  Two structures are
    "chemically equivalent" iff there are zero CHEMISTRY and zero TOPOLOGY
    differences — see spec §8.2 and the equivalence rule.
    """

    NAMING = "naming"
    """Same molecule, different labels.  Vanishes under canonicalization.
    Examples: HSD→HIS, HB1/HB2→HB2/HB3, O1P→OP1, TIP3→HOH, ring-flip swaps.
    """

    TOPOLOGY = "topology"
    """Composition of the model changed.
    Examples: residues added/removed, chains added/removed, gaps opened/closed,
    waters or ions stripped, altlocs resolved.
    """

    CHEMISTRY = "chemistry"
    """Molecular identity changed.
    Examples: mutations, MSE→MET, protonation/tautomer change,
    disulfide formed/broken, ligand replaced.
    """

    GEOMETRY = "geometry"
    """Same atoms, moved.
    Examples: per-residue and global RMSD, coordinate ring flips.
    """


# ---------------------------------------------------------------------------
# Canonical residue
# ---------------------------------------------------------------------------

@dataclass
class CanonicalResidue:
    """Dialect-independent residue identity.

    The triple (parent_name, protonation_state, modification) is the full
    chemical identity.  Do not collapse these into one string — see spec §9.4.

    ``parent_name`` is always the PDB v3 three-letter code for the *canonical*
    parent compound (e.g. HSD, HSE, HSP all map to "HIS").

    ``protonation_state`` captures which tautomer or charge state applies:
      - "delta"   — proton on N-delta (CHARMM HSD / AMBER HID)
      - "epsilon" — proton on N-epsilon (CHARMM HSE / AMBER HIE)
      - "both"    — doubly protonated (CHARMM HSP / AMBER HIP)
      - "protonated"  — generic protonated state (ASH, GLH)
      - "neutral"     — neutral form (LYN)
      - "deprotonated" — negatively charged (CYM)
      - ""        — not applicable (standard residue)

    ``modification`` captures real chemical changes that are NOT pure naming:
      - "selenomethionine" — MSE; Se replaces S (chemistry difference)
      - "disulfide"        — CYX / CYS2; disulfide-bonded cysteine
      - ""                 — standard (no modification)
    """

    chain_id: str
    seq_id: int
    insertion_code: str     # '' when absent
    parent_name: str        # e.g. 'HIS' (always PDB v3)
    protonation_state: str  # e.g. 'delta', 'epsilon', 'both', '' when N/A
    modification: str       # e.g. 'selenomethionine', '' when standard
    dialect_name: str       # original name in file (e.g. 'HSD')


# ---------------------------------------------------------------------------
# Canonical atom
# ---------------------------------------------------------------------------

@dataclass
class CanonicalAtom:
    """Dialect-independent atom identity, plus reporting fields.

    The first six fields mirror AtomKey and form the canonical identity.
    ``dialect_atomname`` and ``element`` are extra fields kept for reporting
    and for the geometry-based prochiral resolver (see spec §9.3 and stereo.py).

    ``canonical_resname`` is the parent residue's PDB v3 name (e.g. "HIS",
    not "HSD").  ``canonical_atomname`` is the PDB v3 atom name after dialect
    mapping — including geometry-based prochiral resolution for CH2 hydrogens.

    ``altloc`` is the alternate-location indicator character, or '' when the
    atom has no alternate location (primary conformation).
    """

    # --- canonical key fields (must match AtomKey exactly) ---
    chain_id: str
    seq_id: int
    insertion_code: str         # '' when absent
    canonical_resname: str      # parent_name from CanonicalResidue
    canonical_atomname: str     # PDB v3 atom name after dialect mapping
    altloc: str                 # '' when absent

    # --- extra fields for reporting / geometry ---
    dialect_atomname: str       # original atom name in file
    element: str                # element symbol, e.g. 'C', 'N', 'O', 'SE'


# ---------------------------------------------------------------------------
# Frozen key for dict lookups (spec §9.1)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AtomKey:
    """Hashable, frozen version of the canonical atom identity.

    Used as the key in all O(1) atom-matching dicts.  The fields are identical
    to the first six fields of CanonicalAtom; keeping them as a separate
    frozen dataclass lets us use it as a dict/set key without exposing the
    reporting fields (dialect_atomname, element).

    Canonical names are always PDB v3 form.  See spec §9.1.
    """

    chain_id: str
    seq_id: int
    insertion_code: str         # '' when absent
    canonical_resname: str      # PDB v3 residue name (e.g. 'HIS')
    canonical_atomname: str     # PDB v3 atom name (e.g. 'ND1', 'HB2')
    altloc: str                 # '' when absent

    @classmethod
    def from_atom(cls, atom: CanonicalAtom) -> "AtomKey":
        """Construct an AtomKey from a CanonicalAtom."""
        return cls(
            chain_id=atom.chain_id,
            seq_id=atom.seq_id,
            insertion_code=atom.insertion_code,
            canonical_resname=atom.canonical_resname,
            canonical_atomname=atom.canonical_atomname,
            altloc=atom.altloc,
        )
