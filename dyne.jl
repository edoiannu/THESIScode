# =============================================================================
# === DYNAMICS SIMULATION OF A SYSTEM SUBJECT TO A CONTINUOUS DYNE-DETECTION ==
# === DAEMONIC ERGOTROPY AND CAPACITY (AND RESPECTIVE MOMENTA) COMPUTATION ====
# === FROM STATE DYNAMICAL EVOLUTION ==========================================
# =============================================================================
#
# Structure of the file:
#   1. Parameters reading from the input file
#   2. Checks on the input parameters and initial state definition
#   3. Output folder, run size check (params.dat) and its writing
#   4. Workers initialization
#   5. Simulation over the chunks of trajectories
#   6. Central momenta reconstruction
#   7. Results printing on files
#
# =============================================================================

# including required libraries
using Distributed   # for parallel computing
using Printf        # for the formatted printing (@printf) of the results

# preliminar control over arguments number
if length(ARGS) < 1 || length(ARGS) > 2
    error("Type decetion type:\n- For homodyne: 'hod' [detection angle]\n- For heterodyne: 'hed'")
end

# reading parameter from terminal
const det_type = ARGS[1]    # 'const' variables cannot be modified anymore
if det_type == "hod"
    const unravelling = det_type * ARGS[2]
    const ϕ_val = parse(Int64, ARGS[2])
    const het_val = false
elseif det_type == "hed"
    const unravelling = det_type
    const ϕ_val = 0         # dummy value: ϕ is irrelevant for heterodyne detection, but must be defined to avoid errors
    const het_val = true
else
    error("Detection type must be homodyne ('hod') or heterodyne ('hed').")
end

# =============================================================================
# 1. PARAMETERS READING
# =============================================================================

# variables initialization
instate = nothing                   # single character variable that indicates the simulation's initial state 
α_val = nothing                     # resonant field intensity over emitting rate value
η_val = nothing                     # detectiong efficiency value
t_f = nothing                       # simulation's final time
deltat = nothing                    # simulation's time step
NUMBER_OF_TRAJECTORIES = nothing    # simulation's number of trajectories
chunk_dim = nothing                 # number of trajectories to evolve simultaneously

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
    if key != "HISTOTIME"
        value = parts[2]
    else
        value = [parse(Float64,parts[i]) for i in 2:nparts]
    end
    if key == "INSTATE"
        # "global" indicates a global variable
        global instate = value
    elseif key == "ALPHA"
        global α_val = parse(Float64, value)
    elseif key == "ETA"
        global η_val = parse(Float64, value)
    elseif key == "CHUNKDIM"
        global chunk_dim = parse(Int64, value)
    elseif key == "HISTOTIME"
        global target_times = value
    end
end

# number of time intervals implied by the input file: it is computed here (and
# not only inside the workers) because it has to be written in params.dat below
# NUMBER_OF_TIMEINTERVALS = Int64(t_f / deltat)

# =============================================================================
# 2. CHECKS ON THE INPUT PARAMETERS AND INITIAL STATE
# =============================================================================

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

# check on chunk dimension
if chunk_dim < nworkers()
    error("Chunk dimension must be larger than the number of workers.")
end

# =============================================================================
# 3. OUTPUT FOLDER AND RUN SIZE CHECK
# =============================================================================

# string that identifies the input simulation's parameters
inputstring = instate * "_eta" * string(η_val) * "_alpha" * string(α_val)
# path where to save the simulation's results
processpath = "results/" * inputstring * "/"
mkpath(processpath)

# --- run size of a possible previous simulation ------------------------------
# params.dat stores the number of trajectories and of time intervals actually
# used to produce the data contained in this folder: it is the file the plotting
# scripts have to read, so that they no longer depend on input.dat (which may
# have been modified after the simulation)
# same reading scheme used above for input.dat
if isfile(joinpath(processpath, "params.dat"))
    for line in eachline(joinpath(processpath, "params.dat"))
        parts = split(line)
        if isempty(line) || startswith(line, "#")
            continue
        end
        key, value = parts
        if key == "FINALT"
            global t_f = parse(Float64, value)
        elseif key == "dt"
            global deltat = parse(Float64, value)
        elseif key == "NTRAJ"
            global NUMBER_OF_TRAJECTORIES = parse(Int64, value)
        end
    end
    global NUMBER_OF_TIMEINTERVALS = Int64(t_f / deltat)
else
    for line in eachline("input.dat")
        # to split line's elements
        parts = split(line)
        # conditions to skip a line
        if isempty(line) || startswith(line, "#")
            continue
        end
        key, value = parts
        if key == "FINALT"
            global t_f = parse(Float64, value)
        elseif key == "dt"
            global deltat = parse(Float64, value)
        elseif key == "NTRAJ"
            global NUMBER_OF_TRAJECTORIES = parse(Int64, value)
        end
    end
    global NUMBER_OF_TIMEINTERVALS = Int64(t_f / deltat)
    # write on processpath the number of trajectories and of time intervals
    open(joinpath(processpath, "params.dat"), "w") do io
        println(io, "# SIMULATION PARAMETERS USED TO PRODUCE THE DATA IN THIS FOLDER")
        println(io, "NTRAJ\t", NUMBER_OF_TRAJECTORIES)
        println(io, "FINALT\t", t_f)
        println(io, "dt\t\t", deltat)
    end
end

# =============================================================================
# 4. WORKERS INITIALIZATION
# =============================================================================

# workers initialization
@everywhere begin
    # libraries inclusion for each worker
    using JLD2  # necessary because @save is called inside each worker
    include("my_library/my_objects.jl")
    
    # constants definition for each core
    α_over_κ = $α_val
    η = $η_val
    heterodyne = $het_val
    ϕ = $ϕ_val
    c = σ_m         # collapse operator
    finalt = $t_f
    dt = $deltat
    
    clean(x; tol = 1e-14) = abs(x) < tol ? 0 : x    # to set at zero "numerical zeros"
    # collapse operator definition
    # if $det_type == "hod"
    const cops = (clean(cos(deg2rad(ϕ))) + 1im * sin(deg2rad(ϕ))) * c
    # println(cops)
    # else
      #  const cops = c
      #  println(cops)
    # end
    
    NUMBER_OF_TIMEINTERVALS = Int64(finalt / dt)
    tlist = range(0, finalt, NUMBER_OF_TIMEINTERVALS + 1)

    # pevolution function definition: it evolves a single trajectory and returns
    # the list of the states visited along it
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
end

# =============================================================================
# 5. SIMULATION
# =============================================================================

println("Dyne detection: initial ", instate, " state, α/κ = ", α_val, ", η = ", η_val, ", ", NUMBER_OF_TIMEINTERVALS, " time intervals and ", NUMBER_OF_TRAJECTORIES, " number of trajectories...")

prog_time = 0                                           # progressive run time
start_time = time()                                     # initial run time
chunk_ind = 0                                           # to count the chunk number
chunk_num = Int64(NUMBER_OF_TRAJECTORIES / chunk_dim)   # number of chunk
target_indices = [findfirst(t -> abs(t - target) < 1e-9, tlist) for target in target_times] # snapshots' time indices
prog_erg_sum = nothing                                          # progressive daemonic ergotropy sum
prog_cap_sum = nothing                                          # progressive daemonic capacity sum
prog_erg_histo = [Float64[] for _ in 1:length(target_times)]    # list to progressively fill with the single states' ergotropy of each target time
prog_cap_histo = [Float64[] for _ in 1:length(target_times)]    # list to progressively fill with the single states' capacity of each target time
# we compute how many trajectories within a chunck are up to each worker
traj_per_worker = div(chunk_dim , nworkers())
# division remainder (it is possible that the number of trajectory per chunk is not a multiple of the number of workers)
rem = chunk_dim % nworkers()

# cycle over the chunks of trajectories
for i in 1:chunk_num
    global chunk_ind += 1
    chunk_start_time = time()  
    # we prepare the groups of trajectories per worker
    # sync: wait for each worker to finish its task
    # lists to fill with the daemonic ergotropy and capacity evolutions of this chunk's trajectories
    chunk_erg_results = Vector{Any}(undef, nworkers())
    chunk_cap_results = Vector{Any}(undef, nworkers())
    # lists to fill with the single states ergotropy and capacity at target times of this chunk's trajectories
    chunk_erg_histo = Vector{Any}(undef, nworkers())
    chunk_cap_histo = Vector{Any}(undef, nworkers())
    @sync begin
        # cycle over workers
        for (w_idx, w) in enumerate(workers())
            # we asign an extra trajectory to the first 'rem' workers
            n_per_worker = traj_per_worker + (w_idx <= rem ? 1 : 0)
            # remotecall assigns a task to a specific worker
            @async begin
                result = remotecall_fetch(w, ρ_0, chunk_ind, w_idx, n_per_worker) do rho, c_id, w_id, n_traj    # variable redefinition
                    # each worker locally executes its sub-chunk
                    local_states = [pevolution(rho) for j in 1:n_per_worker]
                    ρ = [[local_states[k][j] for k in 1:n_traj] for j in 1:NUMBER_OF_TIMEINTERVALS] # ρ[j] = list of states at time step j across this worker's trajectories
                    erg_histo_local = [map(ergotropy, ρ[j]) for j in target_indices]    # list of lists of the single states ergotropy at target times of the trajectories assigned to this worker
                    cap_histo_local = [map(capacity, ρ[j]) for j in target_indices]     # same for the capacity
                    erg_local = map(av_ergotropy, ρ)    # daemonic ergotropy's evolutions of the trajectories assigned to this worker
                    cap_local = map(av_capacity, ρ)     # same for the daemonic capacity
                    # we return the list of the daemonic quantities weighted with respect to the fraction of the chunk's trajectories assigned to this worker
                    return (n_traj .* erg_local ./ chunk_dim, n_traj .* cap_local ./ chunk_dim, erg_histo_local, cap_histo_local)
                end
                chunk_erg_results[w_idx] = result[1]    # each element of this list is the daemonic ergotropy evolution of the w_ixd-th worker
                chunk_cap_results[w_idx] = result[2]    # same for the daemonic capacity
                chunk_erg_histo[w_idx] = result[3]      # each element of this list is the list of the lists single states ergotropy at the given target times of the w_idx-th worker (workers -> target times -> number of trajectory assigned to the worker)
                chunk_cap_histo[w_idx] = result[4]      # same for the capacity
            end
        end
    end
    
    # aggregate results from all workers for this chunk

    erg_chunk = sum(chunk_erg_results)
    cap_chunk = sum(chunk_cap_results)
    
    # cycle over the target times: merge the histograms of all the workers
    for i_t in 1:length(target_times)
        # merge data from all workers for time step i_t in this chunk, then append directly to the global container
        global prog_erg_histo[i_t] = append!(prog_erg_histo[i_t], vcat([chunk_erg_histo[w][i_t] for w in 1:nworkers()]...))
        global prog_cap_histo[i_t] = append!(prog_cap_histo[i_t], vcat([chunk_cap_histo[w][i_t] for w in 1:nworkers()]...))
    end

    # aggregate the results of this chunk to the progressive sums of the daemonic quantities
    if prog_erg_sum === nothing
        global prog_erg_sum = erg_chunk
        # print(erg_chunk)
    else
        global prog_erg_sum += erg_chunk
        # print(erg_chunk)
    end
    if prog_cap_sum === nothing
        global prog_cap_sum = cap_chunk
    else
        global prog_cap_sum += cap_chunk
    end

    # run time of this chunk and progress printing
    chunk_end_time = time()
    global prog_time += chunk_end_time - chunk_start_time
    println(round(Int64(i * chunk_dim) / NUMBER_OF_TRAJECTORIES * 100, digits = 1), "%. Run time: ", round(prog_time, digits = 2), "s.")
end


# =============================================================================
# 6. CENTRAL MOMENTA RECONSTRUCTION
# =============================================================================

# reconstruct central moments from raw moment accumulators E[X^n]
erg_mean = [x[1] for x in prog_erg_sum] ./ chunk_num
cap_mean = [x[1] for x in prog_cap_sum] ./ chunk_num
erg_var  = [x[2] for x in prog_erg_sum] ./ chunk_num - erg_mean .^ 2
cap_var  = [x[2] for x in prog_cap_sum] ./ chunk_num - cap_mean .^ 2
erg_skw  = [x[3] for x in prog_erg_sum] ./ chunk_num - 3 .* erg_mean .* [x[2] for x in prog_erg_sum] ./ chunk_num + 2 .* erg_mean .^ 3
cap_skw  = [x[3] for x in prog_cap_sum] ./ chunk_num - 3 .* cap_mean .* [x[2] for x in prog_cap_sum] ./ chunk_num + 2 .* cap_mean .^ 3

end_time = time()
println("Total run time: ", round(end_time - start_time, digits = 2), "s.")

# =============================================================================
# 7. RESULTS PRINTING
# =============================================================================

# printing results on files
println("Printing results...")
# --- mean values -------------------------------------------------------------
open(processpath * "erg_" * unravelling * ".dat", "w") do io
    println(io, "# NTRAJ\t", NUMBER_OF_TRAJECTORIES)
    for (t, erg) in zip(tlist, erg_mean)
        @printf(io, "%.3f\t%.8f\n", t, erg)
    end
end
open(processpath * "cap_" * unravelling * ".dat", "w") do io
    println(io, "# NTRAJ\t", NUMBER_OF_TRAJECTORIES)
    for (t, cap) in zip(tlist, cap_mean)
        @printf(io, "%.3f\t%.8f\n", t, cap)
    end
end
# --- variances ---------------------------------------------------------------
open(processpath * "var_erg_" * unravelling * ".dat", "w") do io
    println(io, "# NTRAJ\t", NUMBER_OF_TRAJECTORIES)
    for (t, ergvar) in zip(tlist, erg_var)
        @printf(io, "%.3f\t%.8f\n", t, ergvar)
    end
end
open(processpath * "var_cap_" * unravelling * ".dat", "w") do io
    println(io, "# NTRAJ\t", NUMBER_OF_TRAJECTORIES)
    for (t, capvar) in zip(tlist, cap_var)
        @printf(io, "%.3f\t%.8f\n", t, capvar)
    end
end
# --- skewnesses --------------------------------------------------------------
open(processpath * "skw_erg_" * unravelling * ".dat", "w") do io
    println(io, "# NTRAJ\t", NUMBER_OF_TRAJECTORIES)
    for (t, ergskw) in zip(tlist, erg_skw)
        @printf(io, "%.3f\t%.8f\n", t, ergskw)
    end
end
open(processpath * "skw_cap_" * unravelling * ".dat", "w") do io
    println(io, "# NTRAJ\t", NUMBER_OF_TRAJECTORIES)
    for (t, capskw) in zip(tlist, cap_skw)
        @printf(io, "%.3f\t%.8f\n", t, capskw)
    end
end
# --- ergotropy and capacity distributions ------------------------------------
for (i_t, t) in enumerate(target_times)
    open(processpath * "histo_erg_" * unravelling * "_t$(t).dat", "w") do io
        println(io, "# NTRAJ\t", NUMBER_OF_TRAJECTORIES)
        for val in prog_erg_histo[i_t]
            @printf(io, "%.8f\n", val)
        end
    end
    open(processpath * "histo_cap_" * unravelling * "_t$(t).dat", "w") do io
        println(io, "# NTRAJ\t", NUMBER_OF_TRAJECTORIES)
        for val in prog_cap_histo[i_t]
            @printf(io, "%.8f\n", val)
        end
    end
end