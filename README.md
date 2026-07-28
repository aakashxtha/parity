# parity

A CLI tool for auditing and comparing PDB/mmCIF structure files. Useful when you're moving a structure through multiple prep tools (tleap, psfgen, CHARMM-GUI) and need to know if what came out is still the same molecule — or just renamed.

## Install

```bash
git clone https://github.com/aakashxtha/parity.git
cd parity
pip install -e .
```

Requires Python 3.10+. Dependencies (`gemmi`, `typer`, `rich`) are installed automatically.

## Usage

### `parity check` — inspect a single file

```
$ parity check prepped.pdb

  dialect     CHARMM  (confident)
                HSD/HSE/HSP · HB1/HB2 · OT1/OT2 termini · TIP3 water

  contents    2 chains · 487 residues · 7,812 atoms
              1 ligand (LIG) · 2 Zn · 41 waters

  protonation HIS: 9 HSD, 4 HSE, 1 HSP
              CYS: 6 in 3 disulfides, 4 free

  ! gap        chain A 87-93 missing (numbering + CA-CA distance agree)
  ! altloc     3 residues carry alternate locations

  x 2 issues
```

Detects the naming dialect (CHARMM, AMBER, PDB v3, legacy PDB v2), gives you an inventory, and flags gaps, missing atoms, alternate locations, and unknown residues. The most useful output is the mixed-dialect warning — if you concatenated chains from different tools, it'll catch that:

```
  ! dialect    file is not internally consistent
                 chain A          CHARMM  (HSD, HB1/HB2)
                 chain B          AMBER   (HID, HB2/HB3)
```

### `parity diff` — compare two files

```
$ parity diff 1abc_charmm.pdb 1abc_amber.pdb

  chemistry   identical
  topology    identical
  geometry    0.000 A heavy-atom RMSD
  naming      412 atoms across 23 residues

    HSD -> HIS            14 residues
    HB1/HB2 -> HB2/HB3   198 atoms
    O1P/O2P -> OP1/OP2    26 atoms

  = chemically equivalent
```

Differences that are purely naming (HSD→HIS, O1P→OP1, etc.) are separated from real differences. Two structures are **equivalent** only if there are zero chemistry and zero topology differences — naming and geometry don't count.

```
$ parity diff deposited.pdb prepped.pdb

  chemistry   2 differences
  topology    3 differences
  geometry    0.31 A heavy-atom RMSD

  ! MSE 214 -> MET 214          selenomethionine substituted
  ! GOL 401 removed             crystallization additive
  + residues 87-93              built (absent in deposited)

  x structures are not equivalent
```

## Options

**Shared (both commands):**

| Flag | Default | |
|---|---|---|
| `--json` | off | Machine-readable output |
| `--dialect NAME` | auto | Force a dialect: `pdbv3`, `pdbv2`, `amber`, `charmm` |
| `--model N` | 1 | Model to use in multi-model files |
| `--hydrogens` | off | Include hydrogens |
| `--quiet` | off | Exit code only, no output |
| `--no-color` | auto | Disable color |

**`diff` only:**

| Flag | Default | |
|---|---|---|
| `--level` | `all` | Limit to `naming`, `topology`, `chemistry`, `geometry`, or `all` |
| `--rmsd-threshold Å` | `0.0` | Ignore geometry differences below this |
| `--ignore-water` | off | Skip solvent |
| `--ignore-hetatm` | off | Skip non-polymer entities |

## Exit codes

Same convention as `diff(1)` — works in shell pipelines, Snakemake, Nextflow:

| Code | Meaning |
|---|---|
| `0` | No issues (`check`) / equivalent (`diff`) |
| `1` | Issues found (`check`) / differences found (`diff`) |
| `2` | Error (bad file, bad args) |

## Dialects

| Name | Description |
|---|---|
| `pdbv3` | PDB v3 standard — the canonical reference |
| `charmm` | CHARMM36 / psfgen / CHARMM-GUI |
| `amber` | AMBER / tleap / ff14SB |
| `pdbv2` | Legacy PDB naming (pre-2007 remediation) |

Dialect tables are TOML files in `src/parity/data/dialects/`. Adding a new force field is a PR against a data file, no Python needed.

## JSON output

```bash
parity check file.pdb --json
parity diff a.pdb b.pdb --json
```

The `schema_version` field in the output is stable — safe to depend on in scripts.

## License

MIT
