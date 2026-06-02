# === STEADY STATES DYNAMICS SIMULATION OF A SYSTEM SUBJECT TO A CONTINOUS PHOTO-DETECTION ===

# import required libraries and objects
include("my_library/my_objects.jl")
using Printf        # to write on formatted files
using Distributed   # for parallel computing
using JLD2          # to print trajectories on a file

# variables initialization
inputfile = "input.dat"             # input file
unravelling = nothing               # type of unravveling
t_f = nothing                       # evolution final time
dt = nothing                        # time step
α_f = nothing                       # finale value α/κ
NUMBER_OF_ALPHAPOINTS = nothing     # number of α/κ points
NUMBER_OF_TRAJECTORIES = nothing    # number of trajectories to evolve
η = nothing                         # detection efficiency

println("=== STEADY STATES DYNAMICS SIMULATION OF A SYSTEM SUBJECT TO A CONTINOUS PHOTO-DETECTION ===")

# reading from file simulation the remaining parameters
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
    elseif key == "FINALT"
        global t_f = parse(Float64, value)
    elseif key == "dt"
        global dt = parse(Float64, value)
    elseif key == "FINALALPHA"
        global α_f = parse(Float64, value)
    elseif key == "ALPHAPOINTS"
        global NUMBER_OF_ALPHAPOINTS = parse(Int64, value)
    elseif key == "NTRAJ"
        global NUMBER_OF_TRAJECTORIES = parse(Int64, value)
    # elseif key == "CHUNKDIM"
      #  global chunk_dim = parse(Int64, value)
    end
end

η = parse(Float64, ARGS[1])
process = "pd_eta" * string(η)  # process name

# detection efficiency check
if η < 0 || η > 1
    error("The detection efficiency must be between 0 and 1.")
end


dα = Float64(α_f / NUMBER_OF_ALPHAPOINTS)           # α/κ interval width
NUMBER_OF_TIMEINTERVALS = Int64(t_f / dt)           # number of time intervals
tlist = range(0, t_f, NUMBER_OF_TIMEINTERVALS + 1)  # list of time intervals ("+ 1" because it starts with t=0)
αlist = range(0, α_f, NUMBER_OF_ALPHAPOINTS + 1)    # list of α/κ values

ss_ergotropies = []
ss_capacities = []

println("Steady states evolutions (η = ", η, ", ", NUMBER_OF_ALPHAPOINTS, " α/κ points, ", NUMBER_OF_TIMEINTERVALS, " time intervals and ", NUMBER_OF_TRAJECTORIES, " trajectories)...")

function pevolution(α_over_κ)
    # initial state as density matrix (complex in general)
    if instate == "p"
        ρ_0 = ComplexF64[0.00000000 0.00000000 ; 0.00000000 1.00000000] # ground state
        # for the exited state
        # ρ_0 = [1 0 ; 0 0]
    elseif instate == "m"
        ρ_0 = ComplexF64[0.50000000 0.00000000 ; 0.00000000 0.50000000] # maximally mixed state (one half the identity matrix)
    else
        println("The initial state must be pure (p) or maximally mixed (m).")
    end
    ρ_t = ρ_0   # initial state at time t=0
    ρ_tdt = nothing
    for i in tlist
        ρ_tdt = photodet_kraus(HS(α_over_κ), ρ_t, σ_m, η)
        ρ_t = ρ_tdt
    end
    return ρ_tdt
end

prog_time = 0    # progressive run time
start_time = time()
αind = 0

for α in αlist
    global αind += 1
    steadystates = []
    chunk_start_time = time()   # we start counting the execution time of the chunk
    steadystates = pmap(pevolution, [α for j in 1:NUMBER_OF_TRAJECTORIES])
    push!(ss_ergotropies, av_ergotropy(steadystates)[1])
    push!(ss_capacities, av_capacity(steadystates)[1])
    chunk_end_time = time()     # we end conuting the execution time of the chunk
    global prog_time += chunk_end_time - chunk_start_time
    println("α/κ = ", round(α, digits = 5), ", ", round(Int64(αind) / NUMBER_OF_ALPHAPOINTS * 100, digits = 1), "%. Run time: ", round(prog_time, digits = 2), "s.")
end

end_time = time()

# we print the results on a file
println("Printing results...")
open("results/ss_erg_" * process * ".dat", "w") do io
    for (t, erg) in zip(αlist, ss_ergotropies)
        @printf(io, "%.3f\t%.8f\n", t, erg)
    end
end
open("results/ss_cap_" * process * ".dat", "w") do io
    for (t, cap) in zip(αlist, ss_capacities)
        @printf(io, "%.3f\t%.8f\n", t, cap)
    end
end