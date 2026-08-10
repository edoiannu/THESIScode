# =============================================================================
# === STEADY STATES OF THE UNCONDITIONAL DYNAMICS =============================
# === ERGOTROPY, CAPACITY AND ENERGY COMPUTATION ==============================
# === AS A FUNCTION OF THE DRIVING FIELD INTENSITY ============================
# =============================================================================
#
# Structure of the file:
#   1. Parameters reading from the input file
#   2. Checks on the input parameters and initial state definition
#   3. Time grid and output folder
#   4. Simulation over the α/κ values
#   5. Results printing on files
#
# For each α/κ value the unconditional state is evolved up to the final time
# and only its last state is kept: it provides the steady state ergotropy,
# capacity and energy, i.e. the lower and upper bounds between which all the
# daemonic quantities computed by the trajectory simulations must lie.
#
# The unconditional dynamics does not depend on the detection efficiency η nor
# on the number of trajectories, so those parameters are not read here. It does
# not depend on the initial state either, since the master equation admits a
# unique stationary solution: INSTATE is read only to allow a consistency check
# (the same steady state must be obtained from 'p' and from 'm').
#
# =============================================================================

# including required libraries
include("my_library/my_objects.jl")
using Printf    # to write on formatted files

println("=== STEADY STATES OF THE UNCONDITIONAL DYNAMICS ===")

# =============================================================================
# 1. PARAMETERS READING
# =============================================================================

# variables initialization
inputfile = "input.dat"             # name of the file from which we read the simulation's parameters
instate = nothing                   # single character variable describing the evolution initial state {p (pure), m (maximally mixed)}
ρ_0 = nothing                       # density matrix initial state
t_f = nothing                       # evolution final time
dt = nothing                        # time step
α_f = nothing                       # final α/κ value (resonant field intensity over emitting rate value)
NUMBER_OF_ALPHAPOINTS = nothing     # number of α/κ points

# reading system's parameters from the input.dat
for line in eachline(inputfile)
    # to split line's elements
    parts = split(line)
    # conditions to skip a line
    if isempty(line) || length(parts) != 2 || startswith(line, "#")
        continue
    end
    key, value = parts
    if key == "INSTATE"
        # 'global' indicates a global variable assignment
        global instate = value
    elseif key == "FINALT"
        global t_f = parse(Float64, value)
    elseif key == "dt"
        global dt = parse(Float64, value)
    elseif key == "FINALALPHA"
        global α_f = parse(Float64, value)
    elseif key == "ALPHAPOINTS"
        global NUMBER_OF_ALPHAPOINTS = parse(Int64, value)
    end
end

# =============================================================================
# 2. CHECKS ON THE INPUT PARAMETERS AND INITIAL STATE
# =============================================================================

# initial state as density matrix (complex in general)
if instate == "p"
    global ρ_0 = ComplexF64[0.00000000 0.00000000 ; 0.00000000 1.00000000]  # ground state
    # for the exited state
    # ρ_0 = [1 0 ; 0 0]
elseif instate == "m"
    global ρ_0 = ComplexF64[0.50000000 0.00000000 ; 0.00000000 0.50000000]  # maximally mixed state (one half the identity matrix)
else
    error("The initial state must be pure (p) or maximally mixed (m).")
end

# =============================================================================
# 3. TIME GRID AND OUTPUT FOLDER
# =============================================================================

NUMBER_OF_TIMEINTERVALS = Int64(t_f / dt)           # number of time intervals
tlist = range(0, t_f, NUMBER_OF_TIMEINTERVALS + 1)  # list of time intervals ("+ 1" because it starts with t=0)

dα = Float64(α_f / NUMBER_OF_ALPHAPOINTS)           # α/κ interval width
αlist = range(0, α_f, NUMBER_OF_ALPHAPOINTS + 1)    # list of α/κ values

# process name generation
process = "unc"     # process name
# the steady states mix several α/κ values, so they do not belong to a single
# process folder: they are written in the results' root
mkpath("results/")

# =============================================================================
# 4. SIMULATION
# =============================================================================

println("Steady states evolutions (", NUMBER_OF_ALPHAPOINTS, " α/κ points and ", NUMBER_OF_TIMEINTERVALS, " time intervals)...")

# lists to fill with the steady states ergotropy, capacity and energy
ss_ergotropies = Float64[]
ss_capacities = Float64[]
ss_energies = Float64[]

max_residual = 0.0      # largest distance between the last two states of the evolutions
start_time = time()     # initial run time
αind = 0                # to count the α values

# cycle over the α/κ values
for α in αlist
    # unconditional dynamics of the system for this α/κ value
    states = uncond_evo(HS(α), ρ_0, tlist, σ_m)

    # the last state of the evolution is taken as the steady state
    ρ_ss = states[end]

    push!(ss_ergotropies, ergotropy(ρ_ss))
    push!(ss_capacities, capacity(ρ_ss))
    push!(ss_energies, energy(ρ_ss))

    # convergence check: if the final time is long enough the last two states
    # of the evolution must be indistinguishable
    residual = maximum(abs.(ρ_ss .- states[end - 1])) / dt
    global max_residual = max(max_residual, residual)

    # progress printing
    global αind += 1
    println("α/κ = ", round(α, digits = 5), ", ", round(αind / (NUMBER_OF_ALPHAPOINTS + 1) * 100, digits = 1), "%. Run time: ", round(time() - start_time, digits = 2), "s.")
end

println("Total run time: ", round(time() - start_time, digits = 2), "s.")

# =============================================================================
# 5. RESULTS PRINTING
# =============================================================================

# we print the results on a file
println("Printing results...")

# --- steady state ergotropy --------------------------------------------------
open("results/ss_erg_" * process * ".dat", "w") do io
    for (α, erg) in zip(αlist, ss_ergotropies)
        @printf(io, "%.3f\t%.8f\n", α, erg)
    end
end
# --- steady state capacity ---------------------------------------------------
open("results/ss_cap_" * process * ".dat", "w") do io
    for (α, cap) in zip(αlist, ss_capacities)
        @printf(io, "%.3f\t%.8f\n", α, cap)
    end
end
# --- steady state energy -----------------------------------------------------
open("results/ss_en_" * process * ".dat", "w") do io
    for (α, en) in zip(αlist, ss_energies)
        @printf(io, "%.3f\t%.8f\n", α, en)
    end
end