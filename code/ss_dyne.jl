# === STEADY STATES DYNAMICS SIMULATION OF A SYSTEM SUBJECT TO A CONTINIUOUS PHOTO-DETECTION ===

# import required libraries and objects
using Printf        # to write on formatted files
using Distributed   # for parallel computing
using JLD2          # to print trajectories on a file

# preliminar control over arguments number
if length(ARGS) < 1 || length(ARGS) > 2
    error("Type decetion type:\n- For homodyne: 'hod' [detection angle]\n- For heterodyne: 'hed'")
end

# reading parameter from terminal
const det_type = ARGS[1]    # 'const' variables cannot be modified anymore
if det_type == "hod"
    const ϕ_val = parse(Int64, ARGS[2])
    const het_val = false
elseif det_type == "hed"
    const ϕ_val = 0         # dummy value: ϕ is irrelevant for heterodyne detection, but must be defined to avoid errors
    const het_val = true
else
    error("Detection type must be homodyne ('hod') or heterodyne ('hed').")
end

# variables initialization
inputfile = "input.dat"             # name of the file from which we read the simulation's parameters
unravelling = nothing               # type of unravveling
instate = nothing                   # single character variable that indicates the simulation's initial state
ρ_0 = nothing                       # initial state 
η_val = nothing                     # detection efficiency value
t_f = nothing                       # simulation's final time
deltat = nothing                    # simulation's time step
α_f = nothing                       # final α/κ value (resonant field intensity over emitting rate value)
NUMBER_OF_ALPHAPOINTS = nothing     # number of α/κ points
NUMBER_OF_TRAJECTORIES = nothing    # simulation's number of trajectories
chunk_dim = nothing                 # number of trajectories to evolve simultaneously


println("=== STEADY STATES DYNAMICS SIMULATION OF A SYSTEM SUBJECT TO A CONTINOUS HOMODYNE DETECTION ===")

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
    elseif key == "ETA"
        global η_val = parse(Float64, value) 
    elseif key == "FINALT"
        global t_f = parse(Float64, value)
    elseif key == "dt"
        global deltat = parse(Float64, value)
    elseif key == "FINALALPHA"
        global α_f = parse(Float64, value)
    elseif key == "ALPHAPOINTS"
        global NUMBER_OF_ALPHAPOINTS = parse(Int64, value)
    elseif key == "NTRAJ"
        global NUMBER_OF_TRAJECTORIES = parse(Int64, value)
    elseif key == "CHUNKDIM"
        global chunk_dim = parse(Int64, value)
    end
end

# initial state as density matrix (complex in general)
if instate == "p"
    global ρ_0 = ComplexF64[0.00000000 0.00000000 ; 0.00000000 1.00000000] # ground state
    # for the exited state
    # ρ_0 = [1 0 ; 0 0]
elseif instate == "m"
    global ρ_0 = ComplexF64[0.50000000 0.00000000 ; 0.00000000 0.50000000] # maximally mixed state (one half the identity matrix)
else
    error("The initial state must be pure (p) or maximally mixed (m).")
end

# detection efficiency check
if η_val < 0 || η_val > 1
    error("The detection efficiency must be between 0 and 1.")
end

# process name generation
if det_type == "hod"
    process = "hod" * string(ϕ_val) * "_" * string(η_val)  
else
    process = "hed_eta" * string(η_val)
end

# workers initialization
@everywhere begin
    # libraries inclusion for each worker
    include("my_library/my_objects.jl")

    # constants definition for each core
    η = $η_val
    c = σ_m         # collapse operator
    finalt = $t_f
    dt = $deltat
    ρ0 = $ρ_0

    NUMBER_OF_TIMEINTERVALS = Int64(finalt / dt)           # number of time intervals
    tlist = range(0, finalt, NUMBER_OF_TIMEINTERVALS + 1)  # list of time intervals ("+ 1" because it starts with t=0)

    # pevolution function definition
    function pevolution(α_over_κ)
        ρ_t = ρ0   # initial state at time t=0
        ρ_tdt = nothing
        for i in tlist
            ρ_tdt = dyne_kraus(HS(α_over_κ), ρ_t, σ_m, η, het_val)
            ρ_t = ρ_tdt
        end
        return ρ_tdt
    end
end

println("Steady states evolutions (η = ", η, ", ", NUMBER_OF_ALPHAPOINTS, " α/κ points, ", NUMBER_OF_TIMEINTERVALS, " time intervals and ", NUMBER_OF_TRAJECTORIES, " trajectories)...")

dα = Float64(α_f / NUMBER_OF_ALPHAPOINTS)           # α/κ interval width
αlist = range(0, α_f, NUMBER_OF_ALPHAPOINTS + 1)    # list of α/κ values

# lists to fill with steady states ergotropy and capacity
ss_ergotropies = Float64[]
ss_capacities = Float64[]
prog_time = 0           # progressive run time
start_time = time()     # initial run time
αind = 0                                                # to count the α values
chunk_ind = 0                                           # to count the chunk number
chunk_num = Int64(NUMBER_OF_TRAJECTORIES / chunk_dim)   # number of chunk

# we compute how many trajectories within a chunck are up to each worker
traj_per_worker = div(chunk_dim , nworkers())
# division remainder (it is possible that the number of trajectory per chunk is not a multiple of the number of workers)
rem = chunk_dim % nworkers()

for α in αlist
    global αind += 1
    chunk_start_time = time()   # we start counting the execution time of the chunk
    prog_ss_erg_sum = 0         # progressive steady states daemonic ergotropy sum
    prog_ss_cap_sum = 0         # progressive steady states daemonic capacity sum
    for i in 1:chunk_num
        # global chunk_ind += 1
        # lists to fill with the chunk's steady states of each trajectory
        chunk_ss_erg = Vector{Float64}(undef, nworkers())
        chunk_ss_cap = Vector{Float64}(undef, nworkers())
        # chunk_ss = Array{Any}(undef, nworkers())

        # we prepare the groups of trajectories per worker
        # sync: wait for each worker to finish its task
        @sync begin
            # cycle over workers
            for (w_idx, w) in enumerate(workers())
                # we assign an extra trajectory to the first 'rem' workers
                n_per_worker = traj_per_worker + (w_idx <= rem ? 1 : 0)
                # remotecall assigns a task to a specific worker
                @async begin
                    # local steady-states
                    results = remotecall_fetch(w, ρ_0, n_per_worker) do rho, n_traj
                        # each worker locally executes its sub-chunk
                        local_ss = [pevolution(α) for j in 1:n_traj]
                        local_ss_erg = av_ergotropy(local_ss)[1]
                        local_ss_cap = av_capacity(local_ss)[1]
                        # return (n_traj * local_ss_erg / chunk_dim, n_traj * local_ss_cap / chunk_dim)
                        return (local_ss_erg, local_ss_cap)
                        # return local_ss
                    end
                    chunk_ss_erg[w_idx] = results[1]
                    chunk_ss_cap[w_idx] = results[2]
                    # chunk_ss[w_idx] = results
                end
            end
        end

        # aggregate results from all workers for this chunk
        ss_erg_appo = sum(chunk_ss_erg) / nworkers()
        ss_cap_appo = sum(chunk_ss_cap) / nworkers()

        # aggregate the results of this chunk to the progressive sums of the daemonic quantities
        if prog_ss_erg_sum == 0
            prog_ss_erg_sum = ss_erg_appo
        else
            prog_ss_erg_sum += ss_erg_appo
        end
        if prog_ss_cap_sum == 0
            prog_ss_cap_sum = ss_cap_appo
        else
            prog_ss_cap_sum += ss_cap_appo
        end
    end

    push!(ss_ergotropies, prog_ss_erg_sum / chunk_num)
    push!(ss_capacities, prog_ss_cap_sum / chunk_num)
    chunk_end_time = time()     # we end counting the execution time of the chunk
    global prog_time += chunk_end_time - chunk_start_time
    println("α/κ = ", round(α, digits = 5), ", ", round(αind / NUMBER_OF_ALPHAPOINTS * 100, digits = 1), "%. Run time: ", round(prog_time, digits = 2), "s.")
end

end_time = time()
println("Total run time: ", round(end_time - start_time, digits = 2), "s.")

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