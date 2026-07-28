"""check.py — parity check command logic.

Runs structural integrity checks on a single PDB/mmCIF file and returns a
CheckReport dataclass summarising all findings.
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Optional

# ---------------------------------------------------------------------------
# Expected heavy-atom sets for standard amino acids
# ---------------------------------------------------------------------------

EXPECTED_HEAVY_ATOMS: dict[str, set[str]] = {
    'ALA': {'N', 'CA', 'C', 'O', 'CB'},
    'ARG': {'N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'NE', 'CZ', 'NH1', 'NH2'},
    'ASN': {'N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'ND2'},
    'ASP': {'N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'OD2'},
    'CYS': {'N', 'CA', 'C', 'O', 'CB', 'SG'},
    'GLN': {'N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'NE2'},
    'GLU': {'N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'OE2'},
    'GLY': {'N', 'CA', 'C', 'O'},
    'HIS': {'N', 'CA', 'C', 'O', 'CB', 'CG', 'ND1', 'CD2', 'CE1', 'NE2'},
    'ILE': {'N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', 'CD1'},
    'LEU': {'N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2'},
    'LYS': {'N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'CE', 'NZ'},
    'MET': {'N', 'CA', 'C', 'O', 'CB', 'CG', 'SD', 'CE'},
    'PHE': {'N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'},
    'PRO': {'N', 'CA', 'C', 'O', 'CB', 'CG', 'CD'},
    'SER': {'N', 'CA', 'C', 'O', 'CB', 'OG'},
    'THR': {'N', 'CA', 'C', 'O', 'CB', 'OG1', 'CG2'},
    'TRP': {'N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'NE1', 'CE2', 'CE3',
            'CZ2', 'CZ3', 'CH2'},
    'TYR': {'N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OH'},
    'VAL': {'N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2'},
}

# Standard polymer residue names (PDB v3 canonical)
_STANDARD_AA: frozenset[str] = frozenset({
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS',
    'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO',
    'SER', 'THR', 'TRP', 'TYR', 'VAL',
})

# Standard nucleotides
_STANDARD_NA: frozenset[str] = frozenset({
    'DA', 'DC', 'DG', 'DT', 'DI',
    'A', 'C', 'G', 'U', 'I',
})

_STANDARD_POLYMER = _STANDARD_AA | _STANDARD_NA

# Canonical water names
_WATER_NAMES: frozenset[str] = frozenset({'HOH', 'WAT', 'TIP3', 'TIP3P', 'SOL'})

# Elements that are common monatomic ions
_ION_ELEMENTS: frozenset[str] = frozenset({
    'CA', 'ZN', 'MG', 'NA', 'CL', 'K', 'FE', 'MN', 'CU', 'NI',
    'CO', 'CD', 'HG', 'PB', 'BA', 'SR', 'LI', 'RB', 'CS', 'BE',
    'AL', 'CR', 'V', 'MO', 'W', 'PT', 'AU', 'AG',
})


# ---------------------------------------------------------------------------
# CheckReport dataclass
# ---------------------------------------------------------------------------

@dataclass
class CheckReport:
    """Output of a single-file check."""
    source_path: str
    parity_version: str        # from parity.__version__

    # Dialect
    dialect: str
    dialect_confidence: str    # 'confident', 'ambiguous', 'insufficient evidence'
    dialect_alt: Optional[str]      # second candidate if ambiguous
    dialect_evidence: list[str]     # human-readable evidence strings, e.g. ["HSD (×14)"]

    # Per-chain dialect (for mixed-dialect detection)
    chain_dialects: dict[str, tuple[str, str]]  # chain_id -> (dialect, confidence)
    mixed_dialect: bool         # True if chains disagree

    # Inventory
    chain_count: int
    residue_count: int
    atom_count: int
    ligand_names: list[str]     # unique HETATM-class names
    ion_names: list[str]        # monatomic ions found
    water_count: int

    # Protonation census
    his_counts: dict[str, int]   # {'delta': 9, 'epsilon': 4, 'both': 1}
    cys_disulfide: int           # count of CYX/CYS2 residues
    cys_free: int
    has_nonstandard_protonation: bool  # any ASH/GLH/LYN

    # Issues
    chain_breaks: list[dict]     # {'chain', 'from_seq', 'to_seq', 'reason'}
    missing_atoms: list[dict]    # {'chain', 'seq_id', 'resname', 'missing': [...]}
    altloc_residues: list[dict]  # {'chain', 'seq_id', 'resname'}
    unknown_residues: list[dict]
    unknown_atoms: list[dict]
    warnings: list[str]

    # Exit code
    issue_count: int


# ---------------------------------------------------------------------------
# Main check function
# ---------------------------------------------------------------------------

def check_structure(
    path: str | pathlib.Path,
    *,
    dialect_override: Optional[str] = None,
    include_hydrogens: bool = False,
    model_n: int = 1,
) -> CheckReport:
    """Run the check pipeline and return a CheckReport."""
    import numpy as np
    import gemmi

    from parity import __version__
    from parity import dialects as _dialects
    from parity.resolve import _collect_stats

    path = pathlib.Path(path)

    # -----------------------------------------------------------------------
    # 1. Parse structure
    # -----------------------------------------------------------------------
    structure = gemmi.read_structure(str(path))
    model_count = len(structure)
    model_index = model_n - 1

    if model_index < 0 or model_index >= model_count:
        raise ValueError(
            f"model_n={model_n} out of range; file has {model_count} model(s)"
        )

    model = structure[model_index]

    warnings: list[str] = []
    if model_count > 1:
        warnings.append(
            f"File contains {model_count} models; using model {model_n} "
            f"(use --model to select a different one)"
        )

    # -----------------------------------------------------------------------
    # 2. Load dialects and detect
    # -----------------------------------------------------------------------
    all_dialects = _dialects.load_all_dialects()
    stats = _collect_stats(model)

    if dialect_override is not None:
        if dialect_override not in all_dialects:
            raise ValueError(
                f"Unknown dialect '{dialect_override}'. "
                f"Available: {sorted(all_dialects.keys())}"
            )
        dialect_name = dialect_override
        dialect_confidence = "override"
        dialect_alt = None
    else:
        detection = _dialects.detect_dialect(all_dialects, stats)
        dialect_name = detection[0]
        dialect_confidence = detection[2]
        dialect_alt = detection[3] if len(detection) == 4 else None

    # -----------------------------------------------------------------------
    # 3. Build dialect evidence strings
    # -----------------------------------------------------------------------
    dialect_evidence = _build_dialect_evidence(all_dialects[dialect_name], stats)

    # -----------------------------------------------------------------------
    # 4. Per-chain dialect detection
    # -----------------------------------------------------------------------
    chain_dialects: dict[str, tuple[str, str]] = {}
    for chain in model:
        chain_stats = _collect_stats_for_chain(chain)
        chain_detection = _dialects.detect_dialect(all_dialects, chain_stats)
        chain_dialects[chain.name] = (chain_detection[0], chain_detection[2])

    # Check if chains disagree on dialect
    unique_chain_dialects = {v[0] for v in chain_dialects.values()}
    mixed_dialect = len(unique_chain_dialects) > 1

    # -----------------------------------------------------------------------
    # 5. Walk model: build inventory, protonation census, altloc tracking
    # -----------------------------------------------------------------------
    chains_seen: list[str] = []
    chain_ids_seen: set[str] = set()

    # Inventory counters
    total_atoms = 0
    total_residues = 0
    ligand_names_seen: dict[str, int] = {}
    ion_names_seen: dict[str, int] = {}
    water_count = 0

    # Protonation census
    his_counts: dict[str, int] = {'delta': 0, 'epsilon': 0, 'both': 0}
    cys_disulfide = 0
    cys_free = 0
    has_nonstandard_protonation = False

    # For altloc detection: track residues with altloc atoms
    altloc_residue_keys: set[tuple] = set()

    # Per-residue heavy atoms (for missing atom check)
    # Maps (chain_id, seq_id, resname) -> set of canonical atom names seen
    residue_atoms: dict[tuple, set[str]] = {}

    # For chain break detection: chain -> list of (seq_id, CA_position or None)
    chain_residue_list: dict[str, list[tuple[int, Optional[np.ndarray]]]] = {}

    # Track unknown residues from the gemmi walk
    unknown_residues_list: list[dict] = []
    unknown_atoms_list: list[dict] = []

    table = all_dialects[dialect_name]

    for chain in model:
        chain_id = chain.name
        if chain_id not in chain_ids_seen:
            chains_seen.append(chain_id)
            chain_ids_seen.add(chain_id)
        if chain_id not in chain_residue_list:
            chain_residue_list[chain_id] = []

        for residue in chain:
            dialect_resname = residue.name
            seq_id_raw = residue.seqid.num
            try:
                seq_id = int(seq_id_raw)
            except (ValueError, TypeError):
                seq_id = 0
            icode_raw = str(residue.seqid.icode).strip()
            insertion_code = icode_raw if icode_raw and icode_raw != '\x00' else ''

            total_residues += 1

            # Resolve canonical residue name / protonation / modification
            resolved = _dialects.get_residue_canonical(table, dialect_resname)
            if resolved is not None:
                canonical_resname, protonation_state, modification = resolved
            elif dialect_resname in _STANDARD_AA:
                canonical_resname, protonation_state, modification = dialect_resname, '', ''
            else:
                canonical_resname, protonation_state, modification = dialect_resname, '', ''
                # Could be unknown — but we handle classification below

            # Classify residue kind via gemmi entity type and name
            is_water = _is_water(dialect_resname, canonical_resname)
            is_polymer = _is_polymer_residue(residue, dialect_resname, canonical_resname)
            is_ion = (not is_water and not is_polymer
                      and _is_ion_residue(residue))

            if is_water:
                water_count += 1
            elif is_polymer:
                # Standard polymer - included in residue count later
                pass
            elif is_ion:
                # Record ion by residue name (dialect)
                ion_names_seen[dialect_resname] = ion_names_seen.get(dialect_resname, 0) + 1
            else:
                # Ligand (non-water, non-ion HETATM-class)
                ligand_names_seen[dialect_resname] = ligand_names_seen.get(dialect_resname, 0) + 1
                # Flag unknown residues (not in standard polymer and not in table)
                if (resolved is None and dialect_resname not in _STANDARD_AA
                        and dialect_resname not in _STANDARD_NA):
                    unknown_residues_list.append({
                        'chain': chain_id,
                        'seq_id': seq_id,
                        'resname': dialect_resname,
                    })

            # Protonation census
            if canonical_resname == 'HIS' and protonation_state in his_counts:
                his_counts[protonation_state] += 1

            if canonical_resname == 'CYS':
                if modification == 'disulfide':
                    cys_disulfide += 1
                else:
                    cys_free += 1

            if protonation_state in ('protonated', 'neutral', 'deprotonated'):
                has_nonstandard_protonation = True

            # Track CA position for chain break detection (polymer only)
            ca_pos: Optional[np.ndarray] = None
            if is_polymer:
                for atom in residue:
                    if atom.name == 'CA':
                        ca_pos = np.array([atom.pos.x, atom.pos.y, atom.pos.z], dtype=float)
                        break
                chain_residue_list[chain_id].append((seq_id, ca_pos))

            # Count atoms and track altlocs / missing atoms
            heavy_atoms_in_residue: set[str] = set()
            for atom in residue:
                if not include_hydrogens and atom.element.name in ('H', 'D'):
                    continue
                total_atoms += 1

                # Altloc tracking
                altloc_raw = atom.altloc
                altloc = str(altloc_raw) if altloc_raw and altloc_raw != '\x00' else ''
                if altloc:
                    altloc_residue_keys.add((chain_id, seq_id, insertion_code, canonical_resname))

                # Track atom names for missing-atom check (polymer AA only)
                # Use the canonical atom name (via dialect table) for comparison
                if canonical_resname in EXPECTED_HEAVY_ATOMS and atom.element.name not in ('H', 'D'):
                    mapped = _dialects.get_atom_canonical(table, canonical_resname, atom.name)
                    canon_aname = mapped if mapped is not None else atom.name
                    heavy_atoms_in_residue.add(canon_aname)

            # Store heavy atoms found in this residue for missing-atom check
            if canonical_resname in EXPECTED_HEAVY_ATOMS:
                key = (chain_id, seq_id, canonical_resname)
                existing = residue_atoms.get(key, set())
                residue_atoms[key] = existing | heavy_atoms_in_residue

    # -----------------------------------------------------------------------
    # 6. Build altloc_residues list
    # -----------------------------------------------------------------------
    altloc_residues: list[dict] = [
        {'chain': k[0], 'seq_id': k[1], 'resname': k[3]}
        for k in sorted(altloc_residue_keys)
    ]

    # -----------------------------------------------------------------------
    # 7. Missing atoms
    # -----------------------------------------------------------------------
    missing_atoms: list[dict] = []
    for (chain_id, seq_id, canonical_resname), atoms_found in sorted(residue_atoms.items()):
        expected = EXPECTED_HEAVY_ATOMS[canonical_resname]
        missing = expected - atoms_found
        if missing:
            missing_atoms.append({
                'chain': chain_id,
                'seq_id': seq_id,
                'resname': canonical_resname,
                'missing': sorted(missing),
            })

    # -----------------------------------------------------------------------
    # 8. Chain break detection
    # -----------------------------------------------------------------------
    chain_breaks: list[dict] = []
    for chain_id, res_list in chain_residue_list.items():
        if len(res_list) < 2:
            continue
        # Sort by seq_id
        res_list_sorted = sorted(res_list, key=lambda x: x[0])
        for i in range(len(res_list_sorted) - 1):
            seq_curr, ca_curr = res_list_sorted[i]
            seq_next, ca_next = res_list_sorted[i + 1]

            numbering_gap = abs(seq_next - seq_curr) > 1

            ca_distance_gap = False
            if ca_curr is not None and ca_next is not None:
                dist = float(np.linalg.norm(ca_next - ca_curr))
                ca_distance_gap = dist > 4.0

            if numbering_gap or ca_distance_gap:
                if numbering_gap and ca_distance_gap:
                    reason = 'both'
                elif numbering_gap:
                    reason = 'numbering'
                else:
                    reason = 'distance'
                chain_breaks.append({
                    'chain': chain_id,
                    'from_seq': seq_curr,
                    'to_seq': seq_next,
                    'reason': reason,
                })

    # -----------------------------------------------------------------------
    # 9. Compute final counts
    # -----------------------------------------------------------------------
    # residue_count: all residues excluding waters
    # Consistent with spec showing "487 residues" (non-water)
    residue_count = total_residues - water_count

    ligand_names_unique = sorted(ligand_names_seen.keys())
    ion_names_unique = sorted(ion_names_seen.keys())

    issue_count = (
        len(chain_breaks)
        + len(missing_atoms)
        + len(altloc_residues)
        + len(unknown_residues_list)
        + len(unknown_atoms_list)
    )

    return CheckReport(
        source_path=str(path),
        parity_version=__version__,
        dialect=dialect_name,
        dialect_confidence=dialect_confidence,
        dialect_alt=dialect_alt,
        dialect_evidence=dialect_evidence,
        chain_dialects=chain_dialects,
        mixed_dialect=mixed_dialect,
        chain_count=len(chains_seen),
        residue_count=residue_count,
        atom_count=total_atoms,
        ligand_names=ligand_names_unique,
        ion_names=ion_names_unique,
        water_count=water_count,
        his_counts=his_counts,
        cys_disulfide=cys_disulfide,
        cys_free=cys_free,
        has_nonstandard_protonation=has_nonstandard_protonation,
        chain_breaks=chain_breaks,
        missing_atoms=missing_atoms,
        altloc_residues=altloc_residues,
        unknown_residues=unknown_residues_list,
        unknown_atoms=unknown_atoms_list,
        warnings=warnings,
        issue_count=issue_count,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _collect_stats_for_chain(chain) -> dict:
    """Collect structure_stats dict for a single gemmi chain."""
    from collections import Counter
    residue_names: Counter = Counter()
    atom_names: Counter = Counter()
    atom_in_residue: Counter = Counter()

    for residue in chain:
        rname = residue.name
        residue_names[rname] += 1
        for atom in residue:
            atom_names[atom.name] += 1
            atom_in_residue[(rname, atom.name)] += 1

    return {
        'residue_names': residue_names,
        'atom_names': atom_names,
        'atom_in_residue': atom_in_residue,
    }


def _is_water(dialect_resname: str, canonical_resname: str) -> bool:
    """Return True if this residue is a water molecule."""
    return canonical_resname in _WATER_NAMES or dialect_resname in _WATER_NAMES


def _is_polymer_residue(residue, dialect_resname: str, canonical_resname: str) -> bool:
    """Return True if this residue is a standard polymer residue (AA or nucleotide)."""
    return canonical_resname in _STANDARD_POLYMER or dialect_resname in _STANDARD_POLYMER


def _is_ion_residue(residue) -> bool:
    """Return True if this is a monatomic ion (single heavy atom whose element is an ion)."""
    # Count non-H atoms
    heavy_atoms = [a for a in residue if a.element.name not in ('H', 'D')]
    if len(heavy_atoms) == 1:
        elem = heavy_atoms[0].element.name.upper()
        return elem in _ION_ELEMENTS
    return False


def _build_dialect_evidence(table, stats: dict) -> list[str]:
    """Build human-readable evidence strings from dialect markers.

    Returns list like ["HSD (×14)", "TIP3 (×41)"].
    """
    residue_names = stats.get('residue_names', {})
    atom_names = stats.get('atom_names', {})
    atom_in_residue = stats.get('atom_in_residue', {})

    evidence: list[str] = []
    seen_values: set[str] = set()

    for marker in table.markers:
        kind = marker.get('kind', '')
        value = marker.get('value', '')

        if value in seen_values:
            continue

        if kind == 'residue_name':
            count = residue_names.get(value, 0)
            if count > 0:
                evidence.append(f'{value} (×{count})')
                seen_values.add(value)
        elif kind == 'atom_name':
            count = atom_names.get(value, 0)
            if count > 0:
                evidence.append(f'{value} (×{count})')
                seen_values.add(value)
        elif kind == 'atom_in_residue':
            residue = marker.get('residue', '')
            pair = (residue, value)
            count = atom_in_residue.get(pair, 0)
            if count > 0:
                label = f'{residue}:{value}'
                evidence.append(f'{label} (×{count})')
                seen_values.add(value)

    return evidence
