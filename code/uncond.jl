# === SYSTEM UNCONDITIONAL EVOLUTION ===

# including required libraries
include("my_library/my_objects.jl")
using Printf    # to write on formatted files

# variables initialization
inputfile = "input.dat"     # input file
t_f = nothing               # evolution final time
dt = nothing                # time step
instate = nothing           # single character variable describing the evolution initial state {p (pure), m (maximally mixed)}
ρ_0 = nothing                # density matrix initial state
α_over_κ = nothing          # driving field intensity over the system emission rate
η = nothing                 # detection efficiency

for line in eachline(inputfile)
    parts = split(line) # using split() words separated by a space within the argument string become the elements of a list
    # it skips empty lines, controls that there are exactly two elements per line (otherwise it skips the line) and skip the comments
    if isempty(line) || length(parts) != 2 || startswith(line, "#")
        continue
    end
    key, value = parts
    if key == "INSTATE"
    # 'global' indicates a global variable assignment
        global instate = value 
    elseif key == "ALPHA"
        global α_over_κ = parse(Float64, value)
    elseif key == "ETA"
        global η = parse(Float64, value)
    elseif key == "FINALT"
        global t_f = parse(Float64, value)
    elseif key == "dt"
        global dt = parse(Float64, value)
    end
end

# initial state as density matrix (complex in general)
if instate == "p"
    global ρ_0 = ComplexF64[0.00000000 0.00000000 ; 0.00000000 1.00000000]      # ground state
    # for the exited state
    # ρ_0 = [1 0 ; 0 0]
elseif instate == "m"
    global ρ_0 = ComplexF64[0.50000000 0.00000000 ; 0.00000000 0.50000000]  # maximally mixed state (one half the identity matrix)
else
    println("The initial state must be pure (p) or maximally mixed (m).")
end

NUMBER_OF_TIMEINTERVALS = Int64(t_f / dt)           # number of time intervals
tlist = range(0, t_f, NUMBER_OF_TIMEINTERVALS + 1)  # list of time intervals ("+ 1" because it starts with t=0)

println("System evolution (initial ", instate, " state, α/κ = ", α_over_κ, ", η = ", η, " and ", NUMBER_OF_TIMEINTERVALS, " time intervals)...")

# unconditional dynamics of the system
states = uncond_evo(HS(α_over_κ), ρ_0, tlist, σ_m)

# === Ergotropy and capacity computation ===

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

for (i, t) in enumerate(tlist)
    erg = ergotropy(states[i])
    en = energy(states[i])
    push!(erg_results, erg)
    push!(cap_results, capacity(states[i]))
    push!(en_results, en)
    i != 1 ? push!(pw_results, en / t) : continue
    i != 1 ? push!(erg_pw_results, erg / t) : continue 
end

# saving on a file the results
println("Printing results...")

open("results/erg_unc_" * instate * "_alpha" * string(α_over_κ) * ".dat", "w") do io
    for (t, erg) in zip(tlist, erg_results)
        @printf(io, "%.3f\t%.8f\n", t, erg)
    end
end
open("results/cap_unc_" * instate * "_alpha" * string(α_over_κ) * ".dat", "w") do io
    for (t, cap) in zip(tlist, cap_results)
        @printf(io, "%.3f\t%.8f\n", t, cap)
    end
end
open("results/en_unc_" * instate * "_alpha" * string(α_over_κ) * ".dat", "w") do io
    for (t, en) in zip(tlist, en_results)
        @printf(io, "%.3f\t%.8f\n", t, en)
    end
end
open("results/pw_unc_" * instate * "_alpha" * string(α_over_κ) * ".dat", "w") do io
    for (t, pw) in zip(tlist, pw_results)
        @printf(io, "%.3f\t%.8f\n", t, pw)
    end
end
open("results/erg_pw_unc_" * instate * "_alpha" * string(α_over_κ) * ".dat", "w") do io
    for (t, erg_pw) in zip(tlist, erg_pw_results)
        @printf(io, "%.3f\t%.8f\n", t, erg_pw)
    end
end