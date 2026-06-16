# === DAEMONIC ERGOTROPY AND CAPACITY (AND RESPECTIVE MOMENTA) COMPUTATION FROM STATE DYNAMICS ===

using Printf            # to write on formatted files
using Distributed       # for parallel computing
@everywhere begin
    using JLD2          # to load trajectories from file
    include("my_library/my_objects.jl")
end

# preliminar control over arguments number
if length(ARGS) != 1
    error("Type considered unravelling: {pd, hod[detection_angle], hed}")
end

# reading the unravelling from terminal
const unravelling = ARGS[1]

# variables initialization
inputfile = "data_analysis.dat"     # name of the file from which we read the simulation's parameters
instate = nothing                   # single character variable that indicates the simulation's initial state
α_over_κ = nothing                  # resonant field intensity over emitting rate value
η = nothing                         # detection efficiency value
t_f = nothing                       # simulation's final time
dt = nothing                        # simulation's time step
NUMBER_OF_TRAJECTORIES = nothing    # simulation's number of trajectories
chunk_dim = nothing                 # number of trajectories to analyse simultaneously

# reading data analysis parameters from file
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
        global α_over_κ = parse(Float64, value)
    elseif key == "ETA"
        global η = parse(Float64, value)
    elseif key == "FINALT"
        global t_f = parse(Float64, value)
    elseif key == "dt"
        global dt = parse(Float64, value)
    elseif key == "NTRAJ"
        global NUMBER_OF_TRAJECTORIES = parse(Int64, value)
    elseif key == "CHUNKDIM"
        global chunk_dim = parse(Int64, value)
    end
end

NUMBER_OF_TIMEINTERVALS = Int64(t_f / dt)           # number of time intervals
tlist = range(0, t_f, NUMBER_OF_TIMEINTERVALS + 1)  # time grid: N+1 points from 0 to t_f

# process name generation
process = unravelling * "_" * instate * "_eta" * string(η) * "_alpha" * string(α_over_κ)

println("=== DAEMONIC ERGOTROPY AND CAPACITY (AND RESPECTIVE MOMENTA) COMPUTATION FROM STATES DYNAMICS ===")
println("Averaged quantities computation (unravelling: ", unravelling, ", initial ", instate, " state, α/κ = ", α_over_κ, ", η = ", η, ", ", NUMBER_OF_TIMEINTERVALS, " time intervals and ", NUMBER_OF_TRAJECTORIES, " trajectories)...")

prog_erg_sum = nothing                                  # progressive daemonic ergotropy sum
prog_cap_sum = nothing                                  # progressive daemonic capacity sum
prog_time = 0                                           # progressive run time
start_time = time()                                     # initial run time
chunk_ind = 0                                           # to count the chunks
chunk_num = Int64(NUMBER_OF_TRAJECTORIES / chunk_dim)   # total number of chunks

# we compute how many trajectories within a chunk are up to each worker
traj_per_worker = div(chunk_dim, nworkers())
# division remainder (it is possible that the number of trajectory per chunk is not a multiple of the number of workers)
rem = chunk_dim % nworkers()

for i in 1:chunk_num
    global chunk_ind += 1
    chunk_start_time = time()

    # we collect results from each worker file and compute ergotropy and capacity in parallel
    chunk_erg_results = Vector{Any}(undef, nworkers())
    chunk_cap_results = Vector{Any}(undef, nworkers())

    @sync begin
        for (w_idx, w) in enumerate(workers())
            n_per_worker = traj_per_worker + (w_idx <= rem ? 1 : 0) # distribute remainder across first 'rem' workers
            @async begin
                result = remotecall_fetch(w, chunk_ind, w_idx, n_per_worker, NUMBER_OF_TIMEINTERVALS) do c_id, w_id, n_traj, n_intervals
                    # each worker loads its own file
                    local_states = nothing
                    filename = "states/$(process)_chunk$(c_id)_worker$(w_id).jld2"
                    @load filename local_states

                    # each worker computes daemonic ergotropy and capacity on its own trajectories
                    ρ = [[local_states[k][j] for k in 1:n_traj] for j in 1:n_intervals] # ρ[j] = list of states at time step j across this worker's trajectories
                    erg_local = map(av_ergotropy, ρ)    # daemonic ergotropy's evolutions of the trajectories assigned to this worker
                    cap_local = map(av_capacity, ρ)     # same for the daemonic capacity
                    # we return the list of the daemonic quantities weighted with respect to the fraction of the chunk's trajectories assigned to this worker
                    return (n_traj .* erg_local ./ chunk_dim, n_traj .* cap_local ./ chunk_dim)
                end
                chunk_erg_results[w_idx] = result[1]    # each element of this list is the daemonic ergotropy evolution of the w_ixd-th worker
                chunk_cap_results[w_idx] = result[2]    # same for the daemonic capacity
            end
        end
    end

    # aggregate results from all workers for this chunk
    erg_chunk = sum(chunk_erg_results)
    cap_chunk = sum(chunk_cap_results)

    # aggregate the results of this chunk to the progressive sums of the daemonic quantities
    if prog_erg_sum === nothing
        global prog_erg_sum = erg_chunk
    else
        global prog_erg_sum += erg_chunk
    end
    if prog_cap_sum === nothing
        global prog_cap_sum = cap_chunk
    else
        global prog_cap_sum += cap_chunk
    end

    chunk_end_time = time()
    global prog_time += chunk_end_time - chunk_start_time
    println(round(Int64(i * chunk_dim) / NUMBER_OF_TRAJECTORIES * 100, digits = 1), "%. Run time: ", round(prog_time, digits = 2), "s.")
end

# reconstruct central moments from raw moment accumulators E[X^n]
erg_mean = [x[1] for x in prog_erg_sum] ./ chunk_num
cap_mean = [x[1] for x in prog_cap_sum] ./ chunk_num
erg_var  = [x[2] for x in prog_erg_sum] ./ chunk_num - erg_mean .^ 2
cap_var  = [x[2] for x in prog_cap_sum] ./ chunk_num - cap_mean .^ 2
erg_skw  = [x[3] for x in prog_erg_sum] ./ chunk_num - 3 .* erg_mean .* [x[2] for x in prog_erg_sum] + 2 .* erg_mean .^ 3
cap_skw  = [x[3] for x in prog_cap_sum] ./ chunk_num - 3 .* cap_mean .* [x[2] for x in prog_cap_sum] + 2 .* cap_mean .^ 3

end_time = time()
println("Total run time: ", round(end_time - start_time, digits = 2), "s.")

# printing results on files
println("Printing results...")
open("results/erg_" * process * ".dat", "w") do io
    for (t, erg) in zip(tlist, erg_mean)
        @printf(io, "%.3f\t%.8f\n", t, erg)
    end
end
open("results/cap_" * process * ".dat", "w") do io
    for (t, cap) in zip(tlist, cap_mean)
        @printf(io, "%.3f\t%.8f\n", t, cap)
    end
end
open("results/var_erg_" * process * ".dat", "w") do io
    for (t, ergvar) in zip(tlist, erg_var)
        @printf(io, "%.3f\t%.8f\n", t, ergvar)
    end
end
open("results/var_cap_" * process * ".dat", "w") do io
    for (t, capvar) in zip(tlist, cap_var)
        @printf(io, "%.3f\t%.8f\n", t, capvar)
    end
end
open("results/skw_erg_" * process * ".dat", "w") do io
    for (t, ergskw) in zip(tlist, erg_skw)
        @printf(io, "%.3f\t%.8f\n", t, ergskw)
    end
end
open("results/skw_cap_" * process * ".dat", "w") do io
    for (t, capskw) in zip(tlist, cap_skw)
        @printf(io, "%.3f\t%.8f\n", t, capskw)
    end
end