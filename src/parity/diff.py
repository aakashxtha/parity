"""diff.py — Compare two canonical structures and classify all differences.

Public API:
  diff_structures(cs_a, cs_b, ...) -> DiffReport

See PARITY_SPEC.md §8 for the full diff specification, and §8.2 for the
classification taxonomy (naming / topology / chemistry / geometry).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from parity.model import CanonicalAtom, CanonicalResidue, AtomKey, DiffLevel
from parity.resolve import CanonicalStructure
from parity import align as _align

try:
    from parity import __version__ as _parity_version
except ImportError:
    _parity_version = "0.1.0"


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Difference:
    level: DiffLevel
    kind: str           # 'residue_name', 'atom_name', 'residue_added', 'residue_removed',
                        # 'chain_added', 'chain_removed', 'tautomer', 'modification',
                        # 'coordinates', 'ring_flip'
    chain: Optional[str]
    seq_id: Optional[int]
    insertion_code: str
    from_val: Optional[str]     # what it is in file_a (or None if added)
    to_val: Optional[str]       # what it is in file_b (or None if removed)
    atom_name: Optional[str]    # for atom-level differences
    count: int                  # how many atoms this entry represents
    rmsd: Optional[float]       # for geometry differences
    message: str                # human-readable summary


@dataclass
class DiffReport:
    file_a: str
    file_b: str
    parity_version: str

    # Equivalence
    equivalent: bool            # True iff zero chemistry + zero topology diffs

    # Counts by level
    chemistry_count: int
    topology_count: int
    geometry_rmsd: Optional[float]      # heavy-atom global RMSD, or None if not computed
    naming_atom_count: int
    naming_residue_count: int

    # All differences, classified
    differences: list[Difference] = field(default_factory=list)

    # Dialect info for each file
    dialect_a: str = ""
    dialect_b: str = ""

    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Main comparison function
# ---------------------------------------------------------------------------

def diff_structures(
    cs_a: CanonicalStructure,
    cs_b: CanonicalStructure,
    *,
    level: str = 'all',             # 'naming', 'topology', 'chemistry', 'geometry', 'all'
    rmsd_threshold: float = 0.0,
    ignore_water: bool = False,
    ignore_hetatm: bool = False,
) -> DiffReport:
    """Compare two canonical structures and classify all differences.

    Algorithm (spec §8.3):
    1. Fast path: identical chain IDs, seq_ids, parent_names → zip directly.
    2. Chain pairing: call align.pair_chains(); unpaired → TOPOLOGY.
    3. Per chain pair:
       a. align_residues() to get (res_a, res_b) pairs
       b. One None → TOPOLOGY (added/removed residue)
       c. Both present, different parent_name → CHEMISTRY (mutation) unless
          it's a naming variant (same canonical, different dialect_name)
       d. Both present, same parent_name, different protonation_state → CHEMISTRY (tautomer)
       e. Both present, same parent_name, different modification → CHEMISTRY
       f. Both present, same parent_name → atom-level comparison
    4. Atom-level: build canonical_atomname → atom dicts; match by key.
       - Missing atom → TOPOLOGY
       - Coordinate difference > rmsd_threshold → GEOMETRY
    5. Naming differences: dialect_name differs but parent_name is same → NAMING
    6. Global heavy-atom RMSD over all matched atom pairs.
    7. Equivalence = zero CHEMISTRY + zero TOPOLOGY.
    """
    import numpy as np

    differences: list[Difference] = []
    warnings: list[str] = list(cs_a.warnings) + list(cs_b.warnings)

    # --- Build per-chain residue index ---
    def _group_by_chain(
        residues: list[CanonicalResidue],
    ) -> dict[str, list[CanonicalResidue]]:
        d: dict[str, list[CanonicalResidue]] = {}
        for r in residues:
            if ignore_water and r.parent_name in _WATER_NAMES:
                continue
            d.setdefault(r.chain_id, []).append(r)
        return d

    chain_res_a = _group_by_chain(cs_a.residues)
    chain_res_b = _group_by_chain(cs_b.residues)

    # Effective chain lists (filtered by ignore_water/hetatm applied at residue level)
    eff_chains_a = [c for c in cs_a.chains if c in chain_res_a or c not in chain_res_a]
    eff_chains_b = [c for c in cs_b.chains if c in chain_res_b or c not in chain_res_b]

    # Use all original chains but pull residue lists from filtered groups
    chains_a = cs_a.chains
    chains_b = cs_b.chains

    # --- Build atom coordinate lookup: AtomKey -> (x, y, z) ---
    def _build_atom_coord_map(
        cs: CanonicalStructure,
    ) -> dict[AtomKey, tuple[float, float, float]]:
        """We need coordinates; re-read from the original file via gemmi."""
        import gemmi
        import pathlib
        path = pathlib.Path(cs.source_path)
        struct = gemmi.read_structure(str(path))
        model = struct[cs.model_index]
        coord_map: dict[AtomKey, tuple[float, float, float]] = {}
        for chain in model:
            for residue in chain:
                for atom in residue:
                    if atom.element.name in ("H", "D"):
                        continue  # heavy atoms only for RMSD
                    chain_id = chain.name
                    seq_id = int(residue.seqid.num)
                    icode_raw = str(residue.seqid.icode).strip()
                    icode = icode_raw if icode_raw and icode_raw != "\x00" else ""
                    altloc_raw = atom.altloc
                    altloc = str(altloc_raw) if altloc_raw and altloc_raw != "\x00" else ""
                    key = AtomKey(
                        chain_id=chain_id,
                        seq_id=seq_id,
                        insertion_code=icode,
                        canonical_resname="",   # placeholder; filled below
                        canonical_atomname=atom.name,
                        altloc=altloc,
                    )
                    coord_map[(chain_id, seq_id, icode, altloc, atom.name)] = (
                        atom.pos.x, atom.pos.y, atom.pos.z
                    )
        return coord_map  # type: ignore[return-value]

    # Build atom lookup from CanonicalAtom list (no file re-read needed for keys;
    # we do need coordinates though, so re-parse the file).
    # For the self-diff fast path we skip coordinates entirely if same object.
    same_object = cs_a is cs_b

    # Build canonical atom index: AtomKey -> CanonicalAtom
    def _build_atom_index(cs: CanonicalStructure) -> dict[AtomKey, CanonicalAtom]:
        idx: dict[AtomKey, CanonicalAtom] = {}
        for atom in cs.atoms:
            key = AtomKey.from_atom(atom)
            idx[key] = atom
        return idx

    atom_idx_a = _build_atom_index(cs_a)
    atom_idx_b = _build_atom_index(cs_b)

    # Coordinate map: build only when needed (geometry level)
    _coord_a: dict | None = None
    _coord_b: dict | None = None

    def _get_coords_a() -> dict:
        nonlocal _coord_a
        if _coord_a is None:
            _coord_a = _build_coord_map(cs_a)
        return _coord_a

    def _get_coords_b() -> dict:
        nonlocal _coord_b
        if _coord_b is None:
            if same_object:
                _coord_b = _get_coords_a()
            else:
                _coord_b = _build_coord_map(cs_b)
        return _coord_b

    # Paired atom coords for RMSD
    matched_pairs_for_rmsd: list[tuple[np.ndarray, np.ndarray]] = []

    # --- Chain pairing ---
    paired_chains, unpaired_a, unpaired_b = _align.pair_chains(
        chains_a, chains_b,
        cs_a.residues, cs_b.residues,
    )

    # Unpaired chains → TOPOLOGY
    if level in ('topology', 'all'):
        for chain in unpaired_a:
            n_res = len(chain_res_a.get(chain, []))
            differences.append(Difference(
                level=DiffLevel.TOPOLOGY,
                kind='chain_removed',
                chain=chain,
                seq_id=None,
                insertion_code='',
                from_val=chain,
                to_val=None,
                atom_name=None,
                count=n_res,
                rmsd=None,
                message=f"chain {chain} removed (present in A, absent in B)",
            ))
        for chain in unpaired_b:
            n_res = len(chain_res_b.get(chain, []))
            differences.append(Difference(
                level=DiffLevel.TOPOLOGY,
                kind='chain_added',
                chain=chain,
                seq_id=None,
                insertion_code='',
                from_val=None,
                to_val=chain,
                atom_name=None,
                count=n_res,
                rmsd=None,
                message=f"chain {chain} added (absent in A, present in B)",
            ))

    # --- Per-chain comparison ---
    for (chain_a_id, chain_b_id) in paired_chains:
        res_list_a = chain_res_a.get(chain_a_id, [])
        res_list_b = chain_res_b.get(chain_b_id, [])

        aligned = _align.align_residues(
            chain_a_id, chain_b_id, res_list_a, res_list_b
        )

        for (res_a, res_b) in aligned:
            # --- Topology: one side missing ---
            if res_a is None and res_b is not None:
                if level in ('topology', 'all'):
                    differences.append(Difference(
                        level=DiffLevel.TOPOLOGY,
                        kind='residue_added',
                        chain=chain_b_id,
                        seq_id=res_b.seq_id,
                        insertion_code=res_b.insertion_code,
                        from_val=None,
                        to_val=res_b.dialect_name,
                        atom_name=None,
                        count=1,
                        rmsd=None,
                        message=(
                            f"{res_b.parent_name} {res_b.seq_id}{res_b.insertion_code}"
                            f" chain {chain_b_id}: added"
                        ),
                    ))
                continue

            if res_b is None and res_a is not None:
                if level in ('topology', 'all'):
                    differences.append(Difference(
                        level=DiffLevel.TOPOLOGY,
                        kind='residue_removed',
                        chain=chain_a_id,
                        seq_id=res_a.seq_id,
                        insertion_code=res_a.insertion_code,
                        from_val=res_a.dialect_name,
                        to_val=None,
                        atom_name=None,
                        count=1,
                        rmsd=None,
                        message=(
                            f"{res_a.parent_name} {res_a.seq_id}{res_a.insertion_code}"
                            f" chain {chain_a_id}: removed"
                        ),
                    ))
                continue

            # Both present — compare residue identity
            assert res_a is not None and res_b is not None

            # Naming-only difference: dialect_name differs but parent_name is same
            if (
                res_a.parent_name == res_b.parent_name
                and res_a.protonation_state == res_b.protonation_state
                and res_a.modification == res_b.modification
            ):
                # Same chemical identity — check naming
                if (
                    level in ('naming', 'all')
                    and res_a.dialect_name != res_b.dialect_name
                ):
                    differences.append(Difference(
                        level=DiffLevel.NAMING,
                        kind='residue_name',
                        chain=chain_a_id,
                        seq_id=res_a.seq_id,
                        insertion_code=res_a.insertion_code,
                        from_val=res_a.dialect_name,
                        to_val=res_b.dialect_name,
                        atom_name=None,
                        count=1,
                        rmsd=None,
                        message=(
                            f"{res_a.dialect_name} -> {res_b.dialect_name}"
                            f" at {chain_a_id}:{res_a.seq_id}{res_a.insertion_code}"
                        ),
                    ))
                # Fall through to atom-level comparison

            elif res_a.parent_name == res_b.parent_name:
                # Same parent but different protonation/modification → CHEMISTRY
                if level in ('chemistry', 'all'):
                    if res_a.protonation_state != res_b.protonation_state:
                        differences.append(Difference(
                            level=DiffLevel.CHEMISTRY,
                            kind='tautomer',
                            chain=chain_a_id,
                            seq_id=res_a.seq_id,
                            insertion_code=res_a.insertion_code,
                            from_val=res_a.protonation_state or res_a.dialect_name,
                            to_val=res_b.protonation_state or res_b.dialect_name,
                            atom_name=None,
                            count=1,
                            rmsd=None,
                            message=(
                                f"{res_a.parent_name} {res_a.seq_id}{res_a.insertion_code}"
                                f" chain {chain_a_id}: tautomer changed"
                                f" ({res_a.dialect_name} -> {res_b.dialect_name})"
                            ),
                        ))
                    if res_a.modification != res_b.modification:
                        differences.append(Difference(
                            level=DiffLevel.CHEMISTRY,
                            kind='modification',
                            chain=chain_a_id,
                            seq_id=res_a.seq_id,
                            insertion_code=res_a.insertion_code,
                            from_val=res_a.modification or res_a.dialect_name,
                            to_val=res_b.modification or res_b.dialect_name,
                            atom_name=None,
                            count=1,
                            rmsd=None,
                            message=(
                                f"{res_a.parent_name} {res_a.seq_id}{res_a.insertion_code}"
                                f" chain {chain_a_id}: modification changed"
                                f" ({res_a.dialect_name} -> {res_b.dialect_name})"
                            ),
                        ))
                # Fall through to atom-level even if chemistry differs

            else:
                # Different parent_name → CHEMISTRY (mutation or substitution)
                if level in ('chemistry', 'all'):
                    differences.append(Difference(
                        level=DiffLevel.CHEMISTRY,
                        kind='residue_name',
                        chain=chain_a_id,
                        seq_id=res_a.seq_id,
                        insertion_code=res_a.insertion_code,
                        from_val=res_a.parent_name,
                        to_val=res_b.parent_name,
                        atom_name=None,
                        count=1,
                        rmsd=None,
                        message=(
                            f"mutation {res_a.parent_name} -> {res_b.parent_name}"
                            f" at {chain_a_id}:{res_a.seq_id}{res_a.insertion_code}"
                        ),
                    ))
                # Don't compare atoms when residue types differ
                continue

            # --- Atom-level comparison for matched residues ---
            # Build dicts: canonical_atomname -> CanonicalAtom (filtered by altloc: prefer '')
            def _residue_atoms(
                atom_idx: dict[AtomKey, CanonicalAtom],
                chain_id: str,
                seq_id: int,
                icode: str,
                resname: str,
            ) -> dict[str, CanonicalAtom]:
                result: dict[str, CanonicalAtom] = {}
                # Scan the index for atoms in this residue
                # Use a prefix scan approach via the chain/seq_id key
                # Build a secondary index on demand; this is O(n) per residue but
                # correct — atoms list is small per residue.
                for key, atom in atom_idx.items():
                    if (
                        key.chain_id == chain_id
                        and key.seq_id == seq_id
                        and key.insertion_code == icode
                    ):
                        # Prefer primary altloc ('')
                        existing = result.get(key.canonical_atomname)
                        if existing is None or (key.altloc == '' and existing.altloc != ''):
                            result[key.canonical_atomname] = atom
                return result

            atoms_a_res = _residue_atoms(
                atom_idx_a, chain_a_id, res_a.seq_id, res_a.insertion_code, res_a.parent_name
            )
            atoms_b_res = _residue_atoms(
                atom_idx_b, chain_b_id, res_b.seq_id, res_b.insertion_code, res_b.parent_name
            )

            all_atom_names = set(atoms_a_res) | set(atoms_b_res)
            for aname in sorted(all_atom_names):
                atom_a = atoms_a_res.get(aname)
                atom_b = atoms_b_res.get(aname)

                if atom_a is None:
                    # Atom present in B but not A → TOPOLOGY added
                    if level in ('topology', 'all'):
                        differences.append(Difference(
                            level=DiffLevel.TOPOLOGY,
                            kind='atom_added',
                            chain=chain_b_id,
                            seq_id=res_b.seq_id,
                            insertion_code=res_b.insertion_code,
                            from_val=None,
                            to_val=aname,
                            atom_name=aname,
                            count=1,
                            rmsd=None,
                            message=(
                                f"atom {aname} added in {res_b.parent_name}"
                                f" {res_b.seq_id}{res_b.insertion_code}"
                                f" chain {chain_b_id}"
                            ),
                        ))
                    continue

                if atom_b is None:
                    # Atom present in A but not B → TOPOLOGY removed
                    if level in ('topology', 'all'):
                        differences.append(Difference(
                            level=DiffLevel.TOPOLOGY,
                            kind='atom_removed',
                            chain=chain_a_id,
                            seq_id=res_a.seq_id,
                            insertion_code=res_a.insertion_code,
                            from_val=aname,
                            to_val=None,
                            atom_name=aname,
                            count=1,
                            rmsd=None,
                            message=(
                                f"atom {aname} removed from {res_a.parent_name}"
                                f" {res_a.seq_id}{res_a.insertion_code}"
                                f" chain {chain_a_id}"
                            ),
                        ))
                    continue

                # Both atoms present: check dialect atom name (naming diff)
                if (
                    level in ('naming', 'all')
                    and atom_a.dialect_atomname != atom_b.dialect_atomname
                ):
                    differences.append(Difference(
                        level=DiffLevel.NAMING,
                        kind='atom_name',
                        chain=chain_a_id,
                        seq_id=res_a.seq_id,
                        insertion_code=res_a.insertion_code,
                        from_val=atom_a.dialect_atomname,
                        to_val=atom_b.dialect_atomname,
                        atom_name=aname,
                        count=1,
                        rmsd=None,
                        message=(
                            f"atom {atom_a.dialect_atomname} -> {atom_b.dialect_atomname}"
                            f" (canonical: {aname})"
                            f" in {res_a.parent_name} {res_a.seq_id}{res_a.insertion_code}"
                            f" chain {chain_a_id}"
                        ),
                    ))

                # Geometry comparison: need coordinates
                if level in ('geometry', 'all') and not same_object:
                    coord_a_map = _get_coords_a()
                    coord_b_map = _get_coords_b()

                    pos_a = coord_a_map.get(
                        (chain_a_id, res_a.seq_id, res_a.insertion_code,
                         atom_a.altloc, atom_a.dialect_atomname)
                    )
                    pos_b = coord_b_map.get(
                        (chain_b_id, res_b.seq_id, res_b.insertion_code,
                         atom_b.altloc, atom_b.dialect_atomname)
                    )

                    if pos_a is not None and pos_b is not None:
                        pos_a_arr = np.array(pos_a)
                        pos_b_arr = np.array(pos_b)
                        matched_pairs_for_rmsd.append((pos_a_arr, pos_b_arr))

                        dist = float(np.linalg.norm(pos_a_arr - pos_b_arr))
                        if dist > rmsd_threshold:
                            differences.append(Difference(
                                level=DiffLevel.GEOMETRY,
                                kind='coordinates',
                                chain=chain_a_id,
                                seq_id=res_a.seq_id,
                                insertion_code=res_a.insertion_code,
                                from_val=None,
                                to_val=None,
                                atom_name=aname,
                                count=1,
                                rmsd=dist,
                                message=(
                                    f"atom {aname} moved {dist:.3f} A"
                                    f" in {res_a.parent_name} {res_a.seq_id}{res_a.insertion_code}"
                                    f" chain {chain_a_id}"
                                ),
                            ))

    # --- Global RMSD ---
    geometry_rmsd: Optional[float] = None
    if matched_pairs_for_rmsd and not same_object:
        sq_dists = [
            np.sum((a - b) ** 2)
            for a, b in matched_pairs_for_rmsd
        ]
        geometry_rmsd = float(np.sqrt(np.mean(sq_dists)))
    elif same_object:
        geometry_rmsd = 0.0

    # --- Count differences by level ---
    chemistry_count = sum(1 for d in differences if d.level == DiffLevel.CHEMISTRY)
    topology_count = sum(1 for d in differences if d.level == DiffLevel.TOPOLOGY)
    naming_atom_count = sum(
        d.count for d in differences
        if d.level == DiffLevel.NAMING and d.kind == 'atom_name'
    )
    naming_residue_count = sum(
        d.count for d in differences
        if d.level == DiffLevel.NAMING and d.kind == 'residue_name'
    )

    # --- Equivalence rule (spec §8.2) ---
    equivalent = (chemistry_count == 0 and topology_count == 0)

    return DiffReport(
        file_a=cs_a.source_path,
        file_b=cs_b.source_path,
        parity_version=_parity_version,
        equivalent=equivalent,
        chemistry_count=chemistry_count,
        topology_count=topology_count,
        geometry_rmsd=geometry_rmsd,
        naming_atom_count=naming_atom_count,
        naming_residue_count=naming_residue_count,
        differences=differences,
        dialect_a=cs_a.dialect,
        dialect_b=cs_b.dialect,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_WATER_NAMES: frozenset[str] = frozenset({
    "HOH", "WAT", "TIP3", "TIP3P", "SOL", "T3P",
})


def _build_coord_map(cs: CanonicalStructure) -> dict:
    """Re-parse the structure file to extract atom coordinates.

    Returns a dict keyed by (chain_id, seq_id, insertion_code, altloc, dialect_atomname)
    → (x, y, z).  We key by dialect_atomname (the raw name in the file) because
    that's what we store on CanonicalAtom.dialect_atomname and is unique per atom
    within the raw file.
    """
    import gemmi
    import pathlib

    path = pathlib.Path(cs.source_path)
    struct = gemmi.read_structure(str(path))
    model = struct[cs.model_index]

    coord_map: dict[tuple, tuple[float, float, float]] = {}
    for chain in model:
        for residue in chain:
            for atom in residue:
                if atom.element.name in ("H", "D"):
                    continue
                chain_id = chain.name
                seq_id = int(residue.seqid.num)
                icode_raw = str(residue.seqid.icode).strip()
                icode = icode_raw if icode_raw and icode_raw != "\x00" else ""
                altloc_raw = atom.altloc
                altloc = str(altloc_raw) if altloc_raw and altloc_raw != "\x00" else ""
                coord_map[(chain_id, seq_id, icode, altloc, atom.name)] = (
                    atom.pos.x,
                    atom.pos.y,
                    atom.pos.z,
                )
    return coord_map
