"""align.py — Chain pairing and residue-level sequence alignment.

Two public functions:
  pair_chains()     — match chains between two structures by sequence identity
  align_residues()  — align residue sequences within a matched chain pair

See PARITY_SPEC.md §8.3 for the alignment algorithm description.
"""
from __future__ import annotations

from parity.model import CanonicalResidue


# ---------------------------------------------------------------------------
# Chain pairing
# ---------------------------------------------------------------------------

def pair_chains(
    chains_a: list[str],
    chains_b: list[str],
    residues_a: list[CanonicalResidue],
    residues_b: list[CanonicalResidue],
) -> tuple[list[tuple[str, str]], list[str], list[str]]:
    """Pair chains between two structures by sequence identity.

    Returns:
      paired: list of (chain_a, chain_b) pairs, sorted by descending identity
      unpaired_a: chains in A with no pair in B
      unpaired_b: chains in B with no pair in A

    Algorithm:
    1. Fast path: if both files have identical chain ID sets, pair them directly.
    2. Otherwise, for each (chain_a, chain_b) pair, compute sequence identity
       via gemmi.align_string_sequences.
    3. Greedy assignment: take highest-identity pair, remove both chains, repeat.
    4. Leftover chains are unpaired.
    """
    # Fast path: identical chain IDs on both sides
    set_a = set(chains_a)
    set_b = set(chains_b)
    if set_a == set_b and chains_a == chains_b:
        # Same chain ID order — direct pairing
        return (
            [(c, c) for c in chains_a],
            [],
            [],
        )
    if set_a == set_b:
        # Same IDs, different order — still pair by ID
        return (
            [(c, c) for c in chains_a],
            [],
            [],
        )

    # Build per-chain residue lists indexed by chain_id
    def _chain_residues(
        residues: list[CanonicalResidue],
    ) -> dict[str, list[CanonicalResidue]]:
        d: dict[str, list[CanonicalResidue]] = {}
        for r in residues:
            d.setdefault(r.chain_id, []).append(r)
        return d

    chain_res_a = _chain_residues(residues_a)
    chain_res_b = _chain_residues(residues_b)

    # Compute pairwise sequence identity matrix
    import gemmi

    def _seq_identity(res_a: list[CanonicalResidue], res_b: list[CanonicalResidue]) -> float:
        """Compute sequence identity between two residue lists via gemmi alignment."""
        names_a = [r.parent_name for r in res_a]
        names_b = [r.parent_name for r in res_b]
        if not names_a or not names_b:
            return 0.0
        # target_gapo: 0 means gap is allowed at every position in target
        gapo = [0] * len(names_b)
        try:
            result = gemmi.align_string_sequences(names_a, names_b, gapo)
            return result.calculate_identity()
        except Exception:
            # Fallback: exact-match fraction on shorter sequence
            matches = sum(a == b for a, b in zip(names_a, names_b))
            denom = max(len(names_a), len(names_b))
            return matches / denom if denom else 0.0

    # Build all pairwise scores
    scores: list[tuple[float, str, str]] = []
    for ca in chains_a:
        res_a = chain_res_a.get(ca, [])
        for cb in chains_b:
            res_b = chain_res_b.get(cb, [])
            identity = _seq_identity(res_a, res_b)
            scores.append((identity, ca, cb))

    # Sort by descending identity
    scores.sort(key=lambda x: x[0], reverse=True)

    # Greedy assignment
    paired: list[tuple[str, str]] = []
    used_a: set[str] = set()
    used_b: set[str] = set()

    for identity, ca, cb in scores:
        if ca in used_a or cb in used_b:
            continue
        paired.append((ca, cb))
        used_a.add(ca)
        used_b.add(cb)

    # Sort paired list by descending identity for consistent output
    # (already in order since scores list was sorted)
    unpaired_a = [c for c in chains_a if c not in used_a]
    unpaired_b = [c for c in chains_b if c not in used_b]

    return paired, unpaired_a, unpaired_b


# ---------------------------------------------------------------------------
# Residue alignment within a chain pair
# ---------------------------------------------------------------------------

def align_residues(
    chain_id_a: str,
    chain_id_b: str,
    residues_a: list[CanonicalResidue],
    residues_b: list[CanonicalResidue],
) -> list[tuple[CanonicalResidue | None, CanonicalResidue | None]]:
    """Align residue sequences within a chain pair.

    Returns a list of (res_a, res_b) pairs where None means a gap.

    Fast path: if residues match by (seq_id, insertion_code, parent_name) in
    the same order, zip directly.

    Slow path: use gemmi.align_string_sequences to find the alignment, then
    reconstruct the pairing from the CIGAR string.
    """
    if not residues_a and not residues_b:
        return []
    if not residues_a:
        return [(None, r) for r in residues_b]
    if not residues_b:
        return [(r, None) for r in residues_a]

    # Fast path check: same (seq_id, icode, parent_name) in order
    def _fast_path_match() -> bool:
        if len(residues_a) != len(residues_b):
            return False
        for a, b in zip(residues_a, residues_b):
            if (a.seq_id, a.insertion_code, a.parent_name) != (b.seq_id, b.insertion_code, b.parent_name):
                return False
        return True

    if _fast_path_match():
        return [(a, b) for a, b in zip(residues_a, residues_b)]

    # Fallback 1: match by (seq_id, insertion_code) only — same numbering, maybe different names
    def _key(r: CanonicalResidue) -> tuple[int, str]:
        return (r.seq_id, r.insertion_code)

    keys_a = [_key(r) for r in residues_a]
    keys_b = [_key(r) for r in residues_b]

    if sorted(keys_a) == sorted(keys_b) and keys_a == keys_b:
        # Same ordering by seq_id/icode — zip directly
        return [(a, b) for a, b in zip(residues_a, residues_b)]

    # Slow path: use gemmi sequence alignment on parent_names
    import gemmi

    names_a = [r.parent_name for r in residues_a]
    names_b = [r.parent_name for r in residues_b]
    gapo = [0] * len(names_b)

    try:
        result = gemmi.align_string_sequences(names_a, names_b, gapo)
        cigar = result.cigar_str()
        pairs = _parse_cigar_to_pairs(cigar, residues_a, residues_b)
    except Exception:
        # Ultimate fallback: seq_id-based matching
        pairs = _seqid_match(residues_a, residues_b)

    return pairs


def _parse_cigar_to_pairs(
    cigar: str,
    residues_a: list[CanonicalResidue],
    residues_b: list[CanonicalResidue],
) -> list[tuple[CanonicalResidue | None, CanonicalResidue | None]]:
    """Reconstruct residue pairs from a gemmi CIGAR alignment string.

    CIGAR operations (gemmi convention):
      M  — match/mismatch — one residue from each sequence
      I  — insertion in query (A) — residue in A, gap in B
      D  — deletion in query (A) — gap in A, residue in B
    """
    # Parse CIGAR: each op is <count><letter>
    import re
    ops: list[tuple[int, str]] = []
    for m in re.finditer(r'(\d+)([MIDNSHP=X])', cigar):
        count = int(m.group(1))
        op = m.group(2)
        ops.append((count, op))

    pairs: list[tuple[CanonicalResidue | None, CanonicalResidue | None]] = []
    ia = 0
    ib = 0

    for count, op in ops:
        for _ in range(count):
            if op == 'M' or op == '=' or op == 'X':
                # Match or mismatch: consume one from each
                ra = residues_a[ia] if ia < len(residues_a) else None
                rb = residues_b[ib] if ib < len(residues_b) else None
                pairs.append((ra, rb))
                ia += 1
                ib += 1
            elif op == 'I':
                # Insertion in A: residue in A, gap in B
                ra = residues_a[ia] if ia < len(residues_a) else None
                pairs.append((ra, None))
                ia += 1
            elif op == 'D':
                # Deletion in A: gap in A, residue in B
                rb = residues_b[ib] if ib < len(residues_b) else None
                pairs.append((None, rb))
                ib += 1
            # Other ops (N, S, H, P) are typically not emitted by gemmi here

    # Append any remaining residues as gaps
    while ia < len(residues_a):
        pairs.append((residues_a[ia], None))
        ia += 1
    while ib < len(residues_b):
        pairs.append((None, residues_b[ib]))
        ib += 1

    return pairs


def _seqid_match(
    residues_a: list[CanonicalResidue],
    residues_b: list[CanonicalResidue],
) -> list[tuple[CanonicalResidue | None, CanonicalResidue | None]]:
    """Fallback: match residues by (seq_id, insertion_code).

    Residues present in only one file appear as (res, None) or (None, res).
    """
    map_a: dict[tuple[int, str], CanonicalResidue] = {}
    for r in residues_a:
        map_a[(r.seq_id, r.insertion_code)] = r

    map_b: dict[tuple[int, str], CanonicalResidue] = {}
    for r in residues_b:
        map_b[(r.seq_id, r.insertion_code)] = r

    all_keys = sorted(set(map_a) | set(map_b))
    pairs: list[tuple[CanonicalResidue | None, CanonicalResidue | None]] = []
    for key in all_keys:
        pairs.append((map_a.get(key), map_b.get(key)))

    return pairs
