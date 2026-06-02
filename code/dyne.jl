# === DYNAMICS SIMULATION OF A SYSTEM SUBJECT TO A CONTINOUS GENERIC DYNE DETECTION ===

# import required libraries and objects
include("my_library/my_objects.jl")
using Printf        # to write on formatted files
using Distributed   # for parallel computing
using JLD2          # to print trajectories on a file

# variables initialization
inputfile = "input.dat"             # input file
t_f = nothing                       # evolution final time
dt = nothing                        # time step
instate = nothing                   # single character variable describing the evolution initial state {p (pure), m (maximally mixed)}
NUMBER_OF_TRAJECTORIES = nothing    # number of trajectories to evolve
ρ_0 = nothing                       # density matrix initial state
α_over_κ = nothing                  # driving field intensity over the system emission rate
η = nothing                         # detection efficiency
c = σ_m                             # collapse operator
ϕ = nothing                         # angle detection (homodyne detection)
chunk_dim = nothing                 # chunk dimension (number of trajectories to evolve simultaneously)
heterodyne = false                  # heterodyne detection

println("=== DYNAMICS SIMULTAION OF A SYSTEM SUBJECT TO A CONTINOUS GENERIC DYNE DETECTION ===")

# detection type from command line
if ARGS[1] == "hod"
    ϕ = parse(Int64, ARGS[2])
    # initial state, α/κ and η from command line
    instate = ARGS[3]
    α_over_κ = parse(Float64, ARGS[4])
    η = parse(Float64, ARGS[5])
    clean(x; tol = 1e-14) = abs(x) < tol ? 0 : x
    # collapse operator for simulation
    cops = (clean(cos(deg2rad(ϕ))) + 1im * sin(deg2rad(ϕ))) * c
    # process name
    process = "hod" * string(ϕ) * "_" * instate * "_eta" * string(η) * "_alpha" * string(α_over_κ)  
elseif ARGS[1] == "hed"
    heterodyne = true
    cops = c
    # initial state, α/κ and η from command line
    instate = ARGS[2]
    α_over_κ = parse(Float64, ARGS[3])
    η = parse(Float64, ARGS[4])
    # process name
    process = "hed_" * instate * "_eta" * string(η) * "_alpha" * string(α_over_κ)
else
    error("Detection type must be homodyne ('hod') or heterodyne ('hed').")
end

if length(ARGS) != 4 && length(ARGS) != 5
    error("Type simulation parameters separated by a space:\n- For homodyne detection: 'hod' ϕ (detection angle) initial_state {p (pure state), m (maximally mixed state)} α/κ (driving field intensity over the system emission rate) η (detection efficiency).For homodyne detection: 'hod' ϕ (detection angle) initial_state {p (pure state), m (maximally mixed state)} α/κ (driving field intensity over the system emission rate) η (detection efficiency).")
end

# reading from file simulation the remaining parameters
for line in eachline(inputfile)
    parts = split(line) # using split() words separated by a space within the argument string become the elements of a list
    # it skips empty lines, controls that there are exactly two elements per line (otherwise it skips the line) and skip the comments
    if isempty(line) || length(parts) != 2 || startswith(line, "#")
        continue
    end
    key, value = parts
    # 'global' indicates a global variable assignment
    if key == "FINALT"
        global t_f = parse(Float64, value)
    elseif key == "dt"
        global dt = parse(Float64, value)
    elseif key == "NTRAJ"
        global NUMBER_OF_TRAJECTORIES = parse(Int64, value)
    elseif key == "CHUNKDIM"
        global chunk_dim = parse(Int64, value)
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
    error("The initial state must be pure (p) or maximally mixed (m).")
end

# detection efficiency check
if η < 0 || η > 1
    error("The detection efficiency must be between 0 and 1.")
end

NUMBER_OF_TIMEINTERVALS = Int64(t_f / dt)           # number of time intervals
tlist = range(0, t_f, NUMBER_OF_TIMEINTERVALS + 1)  # list of time intervals ("+ 1" because it starts with t=0)

println("System evolution (initial ", instate, " state, α/κ = ", α_over_κ, ", η = ", η, ", ", NUMBER_OF_TIMEINTERVALS, " time intervals and ", NUMBER_OF_TRAJECTORIES, " number of trajectories)...")

prog_time = 0                                           # progressive run time
start_time = time()                                     # initial run time
chunk_ind = 0                                           # chunk index
chunk_num = Int64(NUMBER_OF_TRAJECTORIES / chunk_dim)   # total number of chunk

# function to evolve in parallel different trajectories
function pevolution(ρ_0)
    ρ_t = ρ_0   # initial state at time t=0
    results = [ρ_t]
    for i in tlist
        ρ_tdt = dyne_kraus(HS(α_over_κ), ρ_t, cops, η, heterodyne)
        push!(results, ρ_tdt)
        ρ_t = ρ_tdt
    end
    return results
end

for i in 1:chunk_num
    global chunk_ind += 1
    states = []
    chunk_start_time = time()   # we start counting the execution time of the chunk
    # we map in parallel the stochastic Kraus master equation solver at chunk of initial state with dimension chunk_dim
    states = pmap(pevolution, [ρ_0 for j in 1:chunk_dim])
    @save "states/" * process * "_chunk" * string(chunk_ind) * ".jld2" states # filename states   # we save the states of the trajectories on a file for later use (daemonic ergotropy and capacity computation)
    chunk_end_time = time()     # we end counting the execution time of the chunk
    global prog_time += chunk_end_time - chunk_start_time
    println(round(Int64(i * chunk_dim) / NUMBER_OF_TRAJECTORIES * 100, digits = 1), "%. Run time: ", round(prog_time, digits = 2), "s.")
end

end_time = time()   # final run time

println("Total run time: ", round(end_time - start_time, digits = 2), "s.")
