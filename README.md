# THESIScode

A Julia-based research project for analyzing quantum systems under continuous measurement, focusing on ergotropy, capacity, and daemonic processes in quantum thermodynamics.

## Overview

This repository contains Julia implementations for thesis research on quantum thermodynamics, specifically studying how quantum systems evolve under continuous measurement and how work can be extracted from these systems (ergotropy). The project analyzes both conditional (trajectory-based) and unconditional (ensemble average) dynamics of a driven-dissipative quantum two-level system.

### Key Research Areas

- **Ergotropy**: Maximum work extractable from a quantum state relative to a reference ground state
- **Capacity**: Related measure of available energy in quantum states
- **Daemonic Processes**: Analysis of quantum trajectories subject to continuous measurement (photodetection, homodyne, and heterodyne detection)
- **Unconditional Dynamics**: Ensemble-averaged evolution governed by the master equation
- **Steady-State Analysis**: Long-time behavior of systems under various measurement protocols

## System Model

The simulations study a **driven-dissipative quantum two-level system** with:

- **Parameters**:
  - `α/κ`: Ratio of driving field intensity to spontaneous emission rate
  - `η`: Detection efficiency (0-1)
  - Initial state: Pure (ground state |0⟩) or maximally mixed (I/2)

- **Measurement Types**:
  - **pd** (Photo-detection): Direct photon counting
  - **hod** (Homodyne detection): Measurement at a specific detection angle φ
  - **hed** (Heterodyne detection): Simultaneous measurement of orthogonal quadratures

## Project Structure

```
THESIScode/
├── daemonic_erg_distribution.jl    # Distribution analysis of ergotropy across trajectories at specific times
├── daemonic_ergotropy.jl           # Temporal evolution of mean ergotropy and moments
├── power.jl                         # Ergotropic power: work extraction rate vs time and energy threshold
├── uncond.jl                        # Unconditional (master equation) evolution
├── ss_daemonic_erg.jl              # Steady-state ergotropy across α/κ parameter space
├── ss_uncond.jl                    # Steady-state unconditional dynamics across α/κ parameter space
├── input.dat                        # Configuration file for all simulations
├── my_library/                      # Custom library functions (quantum operators and evolution)
└── results/                         # Output directory for simulation results
```

## Detailed File Descriptions

### Main Simulation Scripts

#### **daemonic_ergotropy.jl**
**Purpose**: Computes temporal evolution of daemonic ergotropy and capacity averaged over many trajectories, including statistical moments (mean, variance, skewness).

**Functionality**:
1. Reads parameters from `input.dat` (initial state, detection efficiency η, driving field α/κ)
2. Launches parallel computation across multiple CPU workers
3. Each worker evolves multiple quantum trajectories using stochastic Kraus operators:
   - For photo-detection: `photodet_kraus()` 
   - For homodyne/heterodyne: `dyne_kraus()`
4. For each trajectory evolution, computes:
   - Ergotropy at each time step: `ergotropy(ρ)`
   - Capacity at each time step: `capacity(ρ)`
   - Raw moments: E[X¹], E[X²], E[X³]
5. Reconstructs central moments (variance, skewness) from raw moments
6. Outputs mean, variance, and skewness time-evolution for ergotropy and capacity

**Outputs** (in `results/{initial_state}_eta{η}_alpha{α}/`):
- `erg_{detection_type}.dat`: Mean ergotropy vs time
- `cap_{detection_type}.dat`: Mean capacity vs time
- `var_erg_{detection_type}.dat`: Variance of ergotropy vs time
- `var_cap_{detection_type}.dat`: Variance of capacity vs time
- `skw_erg_{detection_type}.dat`: Skewness of ergotropy vs time
- `skw_cap_{detection_type}.dat`: Skewness of capacity vs time

#### **daemonic_erg_distribution.jl**
**Purpose**: Captures full statistical distributions of ergotropy and capacity at selected time snapshots.

**Functionality**:
1. Similar parallel trajectory evolution as `daemonic_ergotropy.jl`
2. Instead of computing moments, stores individual ergotropy/capacity values at target times
3. Specified snapshot times in `input.dat` via `HISTOTIME` parameter
4. Generates histograms (raw data) for each snapshot time

**Outputs** (in `results/{initial_state}_eta{η}_alpha{α}/`):
- `histo_erg_{detection_type}_t{T}.dat`: All ergotropy values at time T
- `histo_cap_{detection_type}_t{T}.dat`: All capacity values at time T

These can be used to plot probability distributions and visualize how distributions evolve.

#### **power.jl**
**Purpose**: Analyzes extractable power (work per unit time) and how it varies with energy thresholds.

**Functionality**:
1. Evolves trajectories and computes ergotropic power at each time: `power = ergotropy / time`
2. Tracks energy thresholds: determines the time when each trajectory first crosses an ergotropy threshold
3. Records power value at threshold crossing
4. Computes averages across all trajectories
5. Produces two complementary views:
   - Power evolution with time
   - Power evolution with respect to energy thresholds

**Outputs** (in `results/{initial_state}_eta{η}_alpha{α}/powers/`):
- `avepower_{detection_type}_against_time.dat`: Average power vs time
- `avepower_{detection_type}_against_energy_threshold.dat`: Average power vs ergotropy threshold

#### **uncond.jl**
**Purpose**: Computes unconditional (ensemble-averaged) dynamics using the master equation, providing lower bounds for all daemonic quantities.

**Functionality**:
1. No trajectory sampling - uses deterministic master equation evolution
2. Evolves a single averaged density matrix via `uncond_evo()`
3. Applies Lindblad master equation with spontaneous emission as dissipation
4. Computes for each time step:
   - Ergotropy
   - Capacity
   - Energy (expectation value of system Hamiltonian)
   - Power: energy/time
   - Ergotropic power: ergotropy/time
5. Independent of detection efficiency η (measures full ensemble)

**Outputs** (in `results/`):
- `erg_unc_{initial_state}_alpha{α}.dat`: Unconditional ergotropy vs time
- `cap_unc_{initial_state}_alpha{α}.dat`: Unconditional capacity vs time
- `en_unc_{initial_state}_alpha{α}.dat`: Unconditional energy vs time
- `pw_unc_{initial_state}_alpha{α}.dat`: Unconditional power vs time
- `erg_pw_unc_{initial_state}_alpha{α}.dat`: Unconditional ergotropic power vs time

#### **ss_daemonic_erg.jl**
**Purpose**: Maps steady-state daemonic ergotropy and capacity as functions of the driving field strength α/κ.

**Functionality**:
1. Scans across α/κ values (from 0 to `FINALALPHA` in `ALPHAPOINTS` steps)
2. For each α/κ value:
   - Evolves multiple trajectories for full simulation time
   - Keeps only final state (steady state)
   - Averages ergotropy and capacity across trajectories
3. Shows how measurement feedback affects steady-state work extractability at different driving strengths

**Outputs** (in `results/`):
- `ss_erg_{detection_type}_eta{η}.dat`: Steady-state daemonic ergotropy vs α/κ
- `ss_cap_{detection_type}_eta{η}.dat`: Steady-state daemonic capacity vs α/κ

#### **ss_uncond.jl**
**Purpose**: Maps steady-state unconditional ergotropy, capacity, and energy as functions of α/κ, providing upper bounds.

**Functionality**:
1. Scans α/κ values (same parameter space as `ss_daemonic_erg.jl`)
2. For each α/κ:
   - Evolves unconditional (master equation) state to convergence
   - Checks convergence by comparing successive states
3. Provides reference values for comparison with daemonic results

**Outputs** (in `results/`):
- `ss_erg_unc.dat`: Steady-state unconditional ergotropy vs α/κ
- `ss_cap_unc.dat`: Steady-state unconditional capacity vs α/κ
- `ss_en_unc.dat`: Steady-state unconditional energy vs α/κ

## Configuration File (input.dat)

All simulations read parameters from `input.dat`:

```
# Process Parameters
INSTATE         p              # Initial state: p (pure ground) or m (maximally mixed)
ALPHA           1.0            # Driving field / emission rate (α/κ)
ETA             0.1            # Detection efficiency (0-1)

# Simulation Parameters
FINALT          10             # Final evolution time
dt              0.01           # Time step
NTRAJ           500            # Number of trajectories (for trajectory simulations)

# Parallel Computing
CHUNKDIM        100            # Trajectories per chunk (split across workers)

# Steady-State Scans
FINALALPHA      2              # Maximum α/κ value for steady-state scans
ALPHAPOINTS     50             # Number of α/κ points

# Distribution Snapshots
HISTOTIME       0.5 1.5 2.5 7.5   # Times for ergotropy distribution snapshots

# Power Analysis
NTHRESHOLDS     1000           # Number of energy thresholds
MAXTHRESHOLD    1.0            # Maximum threshold value
```

## Running the Simulations

### Prerequisites
- Julia 1.x or higher
- Quantum computing libraries in `my_library/my_objects.jl` (custom implementations)

### Basic Usage

**Unconditional evolution** (deterministic, single-threaded):
```bash
julia uncond.jl
julia ss_uncond.jl
```

**Conditional (trajectory) evolution** (requires detection type argument):
```bash
# Photo-detection
julia daemonic_ergotropy.jl pd
julia daemonic_erg_distribution.jl pd
julia power.jl pd
julia ss_daemonic_erg.jl pd

# Homodyne detection at 45° phase angle
julia daemonic_ergotropy.jl hod 45
julia ss_daemonic_erg.jl hod 45

# Heterodyne detection
julia daemonic_ergotropy.jl hed
julia ss_daemonic_erg.jl hed
```

**Parallel Execution** (enable multiple CPU cores):
```bash
julia -p auto daemonic_ergotropy.jl pd      # Use all available cores
julia -p 4 daemonic_ergotropy.jl hod 45    # Use 4 cores
```

## Output Analysis

Results are organized hierarchically:

```
results/
├── p_eta0.1_alpha1.0/              # Process folder (one per unique instate, η, α combination)
│   ├── _params.dat                 # Simulation parameters used
│   ├── erg_pd.dat                  # Photo-detection ergotropy mean
│   ├── var_erg_pd.dat              # Photo-detection ergotropy variance
│   ├── skw_erg_pd.dat              # Photo-detection ergotropy skewness
│   ├── histo_erg_pd_t0.5.dat       # Ergotropy distribution at t=0.5
│   ├── powers/
│   │   ├── avepower_pd_against_time.dat
│   │   └── avepower_pd_against_energy_threshold.dat
│   └── [other detection types...]
│
├── ss_erg_pd_eta0.1.dat            # Steady-state daemonic ergotropy vs α/κ
├── ss_cap_pd_eta0.1.dat            # Steady-state daemonic capacity vs α/κ
├── ss_erg_unc.dat                  # Steady-state unconditional ergotropy vs α/κ
├── ss_cap_unc.dat                  # Steady-state unconditional capacity vs α/κ
└── ...
```

## Computational Details

### Parallel Architecture

- Uses Julia's `Distributed` package for multi-core parallelism
- **Chunk-based processing**: Trajectories divided into chunks, each processed by worker pool
- **Work distribution**: Each worker receives `CHUNKDIM/nworkers()` trajectories
- Benefits: Better load balancing and reduced memory per worker

### Quantum Evolution

- **Kraus operators**: Uses completely positive maps for open quantum system evolution
- **Detection types**:
  - Photo-detection: Projects onto detected/not-detected subspaces
  - Homodyne: Measures quadrature at angle φ, collapses accordingly
  - Heterodyne: Measures both quadratures simultaneously

### Numerical Convergence

- Master equation convergence checked in `ss_uncond.jl` via residual between final states
- Time step `dt` must be small enough for accuracy (typically 0.001-0.1)
- Sufficient trajectories needed for statistical convergence (typically 500-10000)

## Physics Interpretation

The project compares three measurement scenarios:

1. **Daemonic (Conditional)**: What quantum systems do when we measure and select specific outcomes
2. **Unconditional (Ensemble)**: Average behavior across all possible measurement outcomes
3. **No Measurement** (implicitly): Reference case with just driving

Key findings typically show:
- Daemonic ergotropy often exceeds unconditional (benefit of feedback)
- Measurement choice (detection type) affects accessible work
- Detection efficiency η reduces daemonic advantage
- Steady-state values depend strongly on α/κ parameter

## License

This project is provided as-is for research purposes.

## Author

**edoiannu** - Thesis research contributor

---

For questions about the simulation methodology, quantum mechanics foundations, or data interpretation, refer to the associated thesis or contact the author.
