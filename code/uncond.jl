# =============================================================================
# === UNCONDITIONAL EVOLUTION OF THE SYSTEM ===================================
# === ERGOTROPY, CAPACITY, ENERGY AND POWERS COMPUTATION ======================
# === FROM STATE DYNAMICAL EVOLUTION ==========================================
# =============================================================================
#
# Structure of the file:
#   1. Parameters reading from the input file
#   2. Checks on the input parameters and initial state definition
#   3. Time grid definition
#   4. Unconditional evolution and computation of the quantities
#   5. Results printing on files
#
# The unconditional dynamics does not depend on the detection efficiency η nor
# on the number of trajectories, so all its parameters are read from input.dat
# and its results are written directly in 'results/'.
#
# =============================================================================

# including required libraries
include("my_library/my_objects.jl")
using Printf    # to write on formatted files

# =============================================================================
# 1. PARAMETERS READING
# =============================================================================

# variables initialization
instate = nothing           # single character variable describing the evolution initial state {p (pure), m (maximally mixed)}
ρ_0 = nothing               # density matrix initial state
α_over_κ = nothing          # driving field intensity over the system emission rate
t_f = nothing               # evolution final time
dt = nothing                # time step

# reading system's parameters from the input.dat
for line in eachline("input.dat")
    # to split line's elements
    parts = split(line)
    nparts = length(parts)
    # conditions to skip a line
    if isempty(line) || startswith(line, "#")
        continue
    end
    key = parts[1]
    value = parts[2]
    if key == "INSTATE"
        # 'global' indicates a global variable assignment
        global instate = value
    elseif key == "ALPHA"
        global α_over_κ = parse(Float64, value)
    elseif key == "FINALT"
        global t_f = parse(Float64, value)
    elseif key == "dt"
        global dt = parse(Float64, value)
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

# string that identifies the initial state and the driving field intensity
inputstring = instate * "_alpha" * string(α_over_κ)
# path where to save the results
processpath = "results/"
mkpath(processpath)

# =============================================================================
# 4. UNCONDITIONAL EVOLUTION AND COMPUTATION OF THE QUANTITIES
# =============================================================================

println("System evolution (initial ", instate, " state, α/κ = ", α_over_κ, " and ", NUMBER_OF_TIMEINTERVALS, " time intervals)...")

# unconditional dynamics of the system
states = uncond_evo(HS(α_over_κ), ρ_0, tlist, σ_m)

println("Ergotropy and capacity computation...")

# list to fill with ergotropy values
erg_results = []
# list to fill with capacity values
cap_results = []
# list to fill with energy values
en_results = []
# list to fill with power values
pw_results = [0.0]
# list to fill with ergotropic power values
erg_pw_results = [0.0]

# cycle over the time intervals
for (i, t) in enumerate(tlist)
    erg = ergotropy(states[i])
    en = energy(states[i])
    push!(erg_results, erg)
    push!(cap_results, capacity(states[i]))
    push!(en_results, en)
    i != 1 ? push!(pw_results, en / t) : continue
    i != 1 ? push!(erg_pw_results, erg / t) : continue
end

# =============================================================================
# 5. RESULTS PRINTING
# =============================================================================

# saving on a file the results
println("Printing results...")

# --- ergotropy and capacity --------------------------------------------------
open(processpath * "erg_unc_" * inputstring * ".dat", "w") do io
    for (t, erg) in zip(tlist, erg_results)
        @printf(io, "%.3f\t%.8f\n", t, erg)
    end
end
open(processpath * "cap_unc_" * inputstring * ".dat", "w") do io
    for (t, cap) in zip(tlist, cap_results)
        @printf(io, "%.3f\t%.8f\n", t, cap)
    end
end
# --- energy ------------------------------------------------------------------
open(processpath * "en_unc_" * inputstring * ".dat", "w") do io
    for (t, en) in zip(tlist, en_results)
        @printf(io, "%.3f\t%.8f\n", t, en)
    end
end
# --- powers ------------------------------------------------------------------
open(processpath * "pw_unc_" * inputstring * ".dat", "w") do io
    for (t, pw) in zip(tlist, pw_results)
        @printf(io, "%.3f\t%.8f\n", t, pw)
    end
end
open(processpath * "erg_pw_unc_" * inputstring * ".dat", "w") do io
    for (t, erg_pw) in zip(tlist, erg_pw_results)
        @printf(io, "%.3f\t%.8f\n", t, erg_pw)
    end
end