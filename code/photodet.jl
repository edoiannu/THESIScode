# === DYNAMICS SIMULATION OF A SYSTEM SUBJECT TO A CONTINUOUS PHOTO-DETECTION ===

# including required libraries
using Distributed   # for parallel computing

# no arguments reading from terminal

# variables initialization
inputfile = "input.dat"             # name of the file from which we read the simulation's paremeters
instate = nothing                   # single character variable that indicates the simulation's initial state 
α_val = nothing                     # resonant field intensity over emitting rate value
η_val = nothing                     # detectiong efficiency value
t_f = nothing                       # simulation's final time
deltat = nothing                    # simulation's time step
NUMBER_OF_TRAJECTORIES = nothing    # simulation's number of trajectories
chunk_dim = nothing                 # number of trajectories to evolve simultaneously

# reading the remaining parameters from file
for line in eachline(inputfile)
    # to split line's elements
    parts = split(line)
    # conditions to skip a line
    if isempty(line) || length(parts) != 2 || startswith(line, "#")
        continue
    end
    key, value = parts
    if key == "INSTATE"
        # "global" indicates a global variable
        global instate = value
    elseif key == "ALPHA"
        global α_val = parse(Float64, value)
    elseif key == "ETA"
        global η_val = parse(Float64, value)
    elseif key == "FINALT"
        global t_f = parse(Float64, value)
    elseif key == "dt"
        global deltat = parse(Float64, value)
    elseif key == "NTRAJ"
        global NUMBER_OF_TRAJECTORIES = parse(Int64, value)
    elseif key == "CHUNKDIM"
        global chunk_dim = parse(Int64, value)
    end
end

# initial state density matrix definition
if instate == "p"
    global ρ_0 = ComplexF64[0.00000000 0.00000000 ; 0.00000000 1.00000000]  # ground state
elseif instate == "m"
    global ρ_0 = ComplexF64[0.50000000 0.00000000 ; 0.00000000 0.50000000]  # maximally mixed state
else
    error("The initial state must be pure (p) or maximally mixed (m).")
end

# check on detection efficiency value
if η_val < 0 || η_val > 1
    error("The detection efficiency must be between 0 and 1.")
end

# string that identifies the input simulation's parameters
inputstring = instate * "_eta" * string(η_val) * "_alpha" * string(α_val)
# string that identifies process name
process = "pd_" * inputstring
# path where to save the states' trajectories
processpath = "states/" * inputstring * "/"
mkpath(processpath)
# write on processpath the number of trajectories
write(joinpath(processpath, "ntraj.dat"), string(NUMBER_OF_TRAJECTORIES), "\n")

# workers initialization
@everywhere begin
    # libraries inclusion for each worker
    using JLD2  # necessary because @save is called inside each worker
    include("my_library/my_objects.jl")
    
    # constants definition for each core
    α_over_κ = $α_val
    η = $η_val
    c = σ_m         # collapse operator
    finalt = $t_f
    dt = $deltat
    
    NUMBER_OF_TIMEINTERVALS = Int64(finalt / dt)
    tlist = range(0, finalt, NUMBER_OF_TIMEINTERVALS + 1)

    # pevolution function definition
    function pevolution(ρ_0)
        ρ_t = ρ_0   # initial state at time t=0
        results = [ρ_t]
        for i in tlist
            ρ_tdt = photodet_kraus(HS(α_over_κ), ρ_t, c, η)
            push!(results, ρ_tdt)
            ρ_t = ρ_tdt
        end
        return results
    end
end

println("System evolution (initial ", instate, " state, α/κ = ", α_val, ", η = ", η_val, ", ", NUMBER_OF_TIMEINTERVALS, " time intervals and ", NUMBER_OF_TRAJECTORIES, " number of trajectories)...")

prog_time = 0                                           # progressive run time
start_time = time()                                     # initial run time
chunk_ind = 0                                           # to count the chunk number
chunk_num = Int64(NUMBER_OF_TRAJECTORIES / chunk_dim)   # number of chunk

# we compute how many trajectories within a chunck are up to each worker
traj_per_worker = div(chunk_dim , nworkers())
# division remainder (it is possible that the number of trajectory per chunk is not a multiple of the number of workers)
rem = chunk_dim % nworkers()

for i in 1:chunk_num
    global chunk_ind += 1
    chunk_start_time = time()  
    # we prepare the groups of trajectories per worker
    # sync: wait for each worker to finish its task
    @sync begin
        # cycle over workers
        for (w_idx, w) in enumerate(workers())
            # we asign an extra trajectory to the first 'rem' workers
            n_per_worker = traj_per_worker + (w_idx <= rem ? 1 : 0)
            # remotecall assigns a task to a specific worker
            @async remotecall_fetch(w, ρ_0, chunk_ind, w_idx) do rho, c_id, w_id    # variable redefinition
                
                # each worker locally executes its sub-chunk
                local_states = [pevolution(rho) for j in 1:n_per_worker]
                
                # each worker saves its local results on disk
                filename = processpath * "$(process)_chunk$(c_id)_worker$(w_id).jld2"
                @save filename local_states
            end
        end
    end
    
    chunk_end_time = time()
    global prog_time += chunk_end_time - chunk_start_time
    println(round(Int64(i * chunk_dim) / NUMBER_OF_TRAJECTORIES * 100, digits = 1), "%. Run time: ", round(prog_time, digits = 2), "s.")
end

end_time = time()
println("Total run time: ", round(end_time - start_time, digits = 2), "s.")
