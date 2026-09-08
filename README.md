# Daemonic Ergotropy in a Continuously Monitored Qubit

Julia code to simulate the dynamical evolution of a qubit (two-level system, driven by a resonant field and subject to decay) and to compute **ergotropy**, **capacity**, **energy** and **ergotropic power**, both for the unconditional evolution (master equation) and for the case of continuous monitoring (conditional quantum trajectories: photodetection, homodyne, heterodyne).

The term "daemonic" refers to the ergotropy averaged over the individual trajectories conditioned by the continuous measurement (in analogy with Maxwell's demon: the information gained from monitoring makes it possible to extract more work than in the unconditional case).

## Physical model

- System: qubit with free Hamiltonian `H0 = ω₀/2 (σ_z + I)`, with `ω₀ = 1` (energies in units of `ω₀`).
- Driving: coherent term `HS(α/κ) = (α/κ) σ_x`, where `α/κ` is the pump field intensity normalized to the emission rate `κ`.
- Dissipation/measurement: collapse operator `σ₋` (spontaneous decay), with a Lindblad equation for the unconditional dynamics.
- The code works in rescaled units (`α → α/κ`, `t → κt`, `√κ c → c`), as stated at the top of `my_objects.jl`.
- Three types of continuous monitoring ("unravelling") are supported:
  - `pd` — photodetection
  - `hod [angle]` — homodyne, with detection angle in degrees
  - `hed` — heterodyne

## File structure

| File | Role |
|---|---|
| `my_objects.jl` | **Core library** (must be included by all scripts, typically as `my_library/my_objects.jl`). Defines the Pauli matrices, the Hamiltonian, the Lindblad/Kraus operators for unconditional and conditional evolution (photodetection and dyne-detection), and the functions `ergotropy`, `capacity`, `energy`, `av_ergotropy`, `av_capacity`. |
| `uncond.jl` | **Unconditional** evolution (master equation): computes ergotropy, capacity, energy and power as a function of time for a given initial state and `α/κ`. Reads parameters from `input.dat`. |
| `ss_uncond.jl` | **Unconditional steady states**: repeats the unconditional evolution over a grid of `α/κ` values and saves only the final state (reference upper/lower bounds for the conditional case). |
| `daemonic_ergotropy.jl` | **Conditional** evolution (Monte Carlo over quantum trajectories, parallelized with `Distributed`): computes mean, variance and skewness of daemonic ergotropy and capacity as a function of time, for a given `α/κ`, `η` and unravelling type. |
| `ss_daemonic_erg.jl` | **Conditional steady states**: like `daemonic_ergotropy.jl`, but evaluates only the final (steady-state) trajectory outcome over a grid of `α/κ` values. |
| `power.jl` | Average **ergotropic power**, computed both as a function of time over a set of conditional trajectories, and as a function of ergotropy thresholds reached by each trajectory in a feedback-assisted charging protocol that halts the battery's charging as soon as a given threshold is reached. |
| `daemonic_erg_distribution.jl` | **Distribution** (histogram) of the ergotropy/capacity values of individual trajectories at given time instants (`HISTOTIME`). |

## Requirements

- **Julia** (≥ 1.6 or so, tested with standard syntax from recent versions)
- Standard packages: `LinearAlgebra`, `Printf`, `Distributed`
- Additional package: `JLD2` (used by the parallel scripts)

`LinearAlgebra`, `Printf` and `Distributed` are part of Julia's standard library.

## Input file: `input.dat`

All scripts read parameters from an `input.dat` file located in the same folder as the script (`KEY value` format; empty lines and lines starting with `#` are ignored). The full set of keys used across the scripts is:

| Key | Meaning | Used by |
|---|---|---|
| `INSTATE` | Initial state: `p` (pure, ground state) or `m` (maximally mixed) | all |
| `ALPHA` | Single value of `α/κ` | `uncond.jl`, `daemonic_ergotropy.jl`, `power.jl`, `daemonic_erg_distribution.jl` |
| `FINALALPHA` | Maximum `α/κ` value in the scan | `ss_uncond.jl`, `ss_daemonic_erg.jl` |
| `ALPHAPOINTS` | Number of points in the `α/κ` scan | `ss_uncond.jl`, `ss_daemonic_erg.jl` |
| `ETA` | Detection efficiency `η ∈ [0,1]` | scripts with conditional dynamics |
| `FINALT` | Final simulation time | all |
| `dt` | Time step | all |
| `NTRAJ` | Number of Monte Carlo trajectories | scripts with conditional dynamics |
| `CHUNKDIM` | Number of trajectories evolved simultaneously per "chunk" (must be ≥ number of workers) | scripts with conditional dynamics |
| `NTHRESHOLDS` | Number of energy thresholds | `power.jl` |
| `MAXTHRESHOLD` | Maximum energy threshold | `power.jl` |
| `HISTOTIME` | List of time instants at which to build the histogram (multiple values on the same line) | `daemonic_erg_distribution.jl` |

Minimal `input.dat` example:

```
# initial state and system parameters
INSTATE p
ALPHA 1.0
ETA 0.8
FINALT 10.0
dt 0.001
NTRAJ 10000
CHUNKDIM 500
```

## Running the scripts

### Unconditional evolution

```bash
julia uncond.jl
```

### Unconditional steady states

```bash
julia ss_uncond.jl
```

### Daemonic ergotropy (conditional dynamics, parallel)

```bash
julia -p N daemonic_ergotropy.jl pd            # photodetection
julia -p N daemonic_ergotropy.jl hod 45        # homodyne, 45° angle
julia -p N daemonic_ergotropy.jl hed           # heterodyne
```

where `N` is the number of Julia workers to start (`-p N`). The same command-line argument scheme (`pd`, `hod <angle>`, `hed`) applies to `ss_daemonic_erg.jl`, `power.jl` and `daemonic_erg_distribution.jl`.

### Daemonic steady states

```bash
julia -p N ss_daemonic_erg.jl pd
```

### Ergotropic power

```bash
julia -p N power.jl hod 0
```

### Daemonic ergotropy distribution

```bash
julia -p N daemonic_erg_distribution.jl hed
```

## Output

Results are written to `results/`, in subfolders named according to the initial state, `η` and `α/κ` (e.g. `results/p_eta0.8_alpha1.0/`). The produced `.dat` files contain tab-separated columns (time/threshold/α and value); among them:

- `erg_unc_*.dat`, `cap_unc_*.dat`, `en_unc_*.dat`, `pw_unc_*.dat`, `erg_pw_unc_*.dat` — unconditional evolution
- `ss_erg_unc.dat`, `ss_cap_unc.dat`, `ss_en_unc.dat` — unconditional steady states
- `erg_<unravelling>.dat`, `cap_<unravelling>.dat`, `var_erg_*.dat`, `var_cap_*.dat`, `skw_erg_*.dat`, `skw_cap_*.dat` — mean, variance and skewness of daemonic ergotropy/capacity
- `ss_erg_<unravelling>_eta<η>.dat`, `ss_cap_<unravelling>_eta<η>.dat` — daemonic steady states as a function of `α/κ`
- `avepower_<unravelling>_against_time.dat`, `avepower_<unravelling>_against_energy_threshold.dat` — average ergotropic power
- `histo_erg_<unravelling>_t<t>.dat`, `histo_cap_<unravelling>_t<t>.dat` — samples for the ergotropy/capacity histograms at a given time instant

Each process folder also contains a `_params.dat` file that records the parameters actually used to generate the data (number of trajectories, final time, time step), so that plotting scripts do not depend on later modifications of `input.dat`.

## Notes

- The parallel scripts reuse existing data: if a process folder already exists with a `_params.dat` file, the run size (`FINALT`, `dt`, `NTRAJ`) is read from there instead of from `input.dat`, to ensure consistency with the data already present.
- The number of trajectories (`NTRAJ`) must be a multiple of `CHUNKDIM`, and `CHUNKDIM` must be greater than or equal to the number of active workers.
