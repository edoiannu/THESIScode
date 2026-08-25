# =============================================================================
# === ERGOTROPIC POWER OF A SYSTEM SUBJECT TO A CONTINUOUS MEASUREMENT ========
# === AVERAGE POWER EVOLUTION AGAINST TIME AND AGAINST ENERGY THRESHOLD =======
# === FROM STATE DYNAMICAL EVOLUTION ==========================================
# =============================================================================
#
# Structure of the file:
#   1. Parameters reading from the input file
#   2. Checks on the input parameters and initial state definition
#   3. Output folder, run size check (params.dat) and its writing
#   4. Workers initialization
#   5. Simulation over the chunks of trajectories
#   6. Averages computation
#   7. Results printing on files
#
# =============================================================================

# including required libraries
using Printf        # to write on formatted files
using Distributed   # for parallel computing

# preliminar control over arguments number
if length(ARGS) < 1 || length(ARGS) > 2
    error("Type considered unravelling: {pd, hod [detection_angle], hed}")
end

# reading the unravelling from terminal
global det_type = ARGS[1]
if det_type == "pd"
    const unravelling = det_type
    const ϕ_val = 0     # dummy value: ϕ is irrelevant for photo.detection and heterodyne detection, but must be defined to avoid errors
    const het_val = false
elseif det_type == "hod"
    const ϕ_val = parse(Int64, ARGS[2])
    const unravelling = det_type * ARGS[2]
    const het_val = false
elseif det_type == "hed"
    const ϕ_val = 0    
    const unravelling = det_type     
    const het_val = true
else
    error("Detection type must be photo-detection ('pd'), homodyne ('hod') or heterodyne ('hed').")
end

# =============================================================================
# 1. PARAMETERS READING
# =============================================================================

# variables initialization
inputfile = joinpath(@__DIR__, "input.dat")
instate = nothing                   # single character variables that indicates the simulation's initial state
α_val = nothing                     # driving field intensity over the system's emitting rate
η_val = nothing                     # detection efficiency value
t_f = nothing                       # simulation's final time
deltat = nothing                    # simulation's time step
NUMBER_OF_TRAJECTORIES = nothing    # simulation's number of trajectory
chunk_dim = nothing                 # (for parallel computing) number of trajectory to evolve simultaneously
Nthresholds = nothing               # number of energy thresholds
MAXthreshold = nothing              # maximum value of energy threshold

# reading system's parameters from the input.dat
for line in eachline(inputfile)
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
        # "global" indicates a global variable
        global instate = value
    elseif key == "ALPHA"
        global α_val = parse(Float64, value)
    elseif key == "ETA"
        global η_val = parse(Float64, value)
    elseif key == "CHUNKDIM"
        global chunk_dim = parse(Int64, value)
    elseif key == "NTHRESHOLDS"
        global Nthresholds = parse(Int64, value)
    elseif key == "MAXTHRESHOLD"
        global MAXthreshold = parse(Float64, value)
    end
end

# the run size (FINALT, dt and NTRAJ) is not read here: it is read in section 3,
# either from params.dat (already existing process) or from input.dat (new one)

# =============================================================================
# 2. CHECKS ON THE INPUT PARAMETERS AND INITIAL STATE
# =============================================================================

# initial state density matrix definition
if instate == "p"
    global ρ_0 = ComplexF64[0.0 0.0 ; 0.0 1.0]  # ground state
elseif instate == "m"
    global ρ_0 = ComplexF64[0.5 0.0 ; 0.0 0.5]  # maximally mixed state
else
    error("The initial state must be pure (p) or maximally mixed (m).")
end

# check on detection efficiency value
if η_val < 0 || η_val > 1
    error("The detection efficiency must be between 0 and 1.")
end

# =============================================================================
# 3. OUTPUT FOLDER AND RUN SIZE CHECK
# =============================================================================

# string that identifies the input simulation's parameters
inputstring = instate * "_eta" * string(η_val) * "_alpha" * string(α_val)
# path where to save the simulation's results
processpath = "results/" * inputstring * "/powers/"
mkpath(processpath)

# --- run size of a possible previous simulation ------------------------------
# params.dat stores the number of trajectories, the final time and the time step
# actually used to produce the data contained in this folder: it is the file the
# plotting scripts have to read, so that they no longer depend on input.dat
# (which may have been modified after the simulation)
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
    # write on processpath the number of trajectories, the final time and the time step
    open(joinpath(processpath, "params.dat"), "w") do io
        println(io, "# SIMULATION PARAMETERS USED TO PRODUCE THE DATA IN THIS FOLDER")
        println(io, "NTRAJ\t", NUMBER_OF_TRAJECTORIES)
        println(io, "FINALT\t", t_f)
        println(io, "dt\t\t", deltat)
    end
end

# energy threshold values
# common variables to all workers
global Elimit = range(0.0, MAXthreshold, Nthresholds)
# global Nthresholds = length(Elimit)

# =============================================================================
# 4. WORKERS INITIALIZATION
# =============================================================================

@everywhere begin
    # libraries inclusion for each worker
    include("my_library/my_objects.jl")

    # constants definition for each core
    α_over_κ = $α_val
    η = $η_val
    heterodyne = $het_val
    ϕ = $ϕ_val
    c = σ_m         # collapse operator
    finalt = $t_f
    dt = $deltat

    NUMBER_OF_TIMEINTERVALS = Int64(finalt / dt)
    tlist = range(0, finalt, NUMBER_OF_TIMEINTERVALS + 1)

    clean(x; tol = 1e-14) = abs(x) < tol ? 0 : x    # to set at zero "numerical zeros"
    # collapse operator definition
    if $det_type == "hod"
        const cops = (clean(cos(deg2rad(ϕ))) + 1im * sin(deg2rad(ϕ))) * c
    else
        const cops = c
    end

    # single trajectory power evolution function definition
    function power_evolution(ρ_0)
        Eidx = 1
        ρ_t = ρ_0   # initial state at time t=0
        power_ev = Vector{Float64}(undef, NUMBER_OF_TIMEINTERVALS)      # power evolution with time
        thr_powers  = zeros(Float64, $Nthresholds)                      # power evolution with thresholds (initialized with Nthreshold zeros)
        thr_reached = falses($Nthresholds)                              # which threshold have been reached from this trajectory
        thr_powers[Eidx] = 0.0
        thr_reached[Eidx] = true
        for i in 1:NUMBER_OF_TIMEINTERVALS
            t = tlist[i]
            ρ_tdt = ($det_type == "pd" ? photodet_kraus(HS(α_over_κ), ρ_t, cops, η) : dyne_kraus(HS(α_over_κ), ρ_t, cops, η, heterodyne))
            erg = ergotropy(ρ_tdt)      # ergotropy for the state at time t + dt
            power = erg / (t + dt)      # ergotropic power for the state at time t + dt
            # updating power's evolution
            power_ev[i] = power
            # updating threshold powers
            for j in (Eidx+1):$Nthresholds
                if erg >= $Elimit[j]
                    thr_powers[j] = power
                    thr_reached[j] = true
                    Eidx = j
                else
                    break
                end
            end
            ρ_t = ρ_tdt
        end
        for i in 1:$Nthresholds
            if !thr_reached[i]
                thr_powers[i] = power_ev[end]
            end
        end
        return power_ev, thr_powers
    end

    # function that for each time interval accumulate the evolutions' power values for a given group or trajectories
    function accumulate_power_evolution(results)
        Ntraj = length(results)
        sum_power_time = zeros(Float64, NUMBER_OF_TIMEINTERVALS)
        sum_power_thre = zeros(Float64, $Nthresholds)
        for j in 1:Ntraj
            sum_power_time .+= results[j][1]
            sum_power_thre .+= results[j][2]
        end
        return sum_power_time, sum_power_thre
    end
end

# =============================================================================
# 5. SIMULATION
# =============================================================================

println("System's power evolution (initial ", instate, " state, α/κ = ", α_val, ", η = ", η_val, ", ", NUMBER_OF_TIMEINTERVALS, " time intervals and ", NUMBER_OF_TRAJECTORIES, " number of trajectories)...")

prog_time = 0                                           # progressive run time
start_time = time()                                     # initial run time
chunk_ind = 0                                           # to count the chunk number
chunk_num = Int64(NUMBER_OF_TRAJECTORIES / chunk_dim)   # number of chunk

# we compute how many trajectories within a chunck are up to each worker
traj_per_worker = div(chunk_dim , nworkers())
# division remainder (it is possible that the number of trajectory per chunk is not a multiple of the number of workers)
rem = chunk_dim % nworkers()

# vectors that, for each time interval or energy threshold, respectively compute the total progressive powers
sum_power_time_tot = zeros(Float64, NUMBER_OF_TIMEINTERVALS)
sum_power_thre_tot = zeros(Float64, Nthresholds)

# cycle over the chunks of trajectories
for i in 1:chunk_num
    chunk_start_time = time()
    # vectors that accumulate the results of each worker of a chunk
    # workers -> (accumulated power time evolution/accumulated power threshold evolution/number of trajectories that reach the threshold within a worker)
    chunk_powers_time = Vector{Any}(undef, nworkers())
    chunk_powers_thre = Vector{Any}(undef, nworkers())
    # chunk_counts_thre = Vector{Any}(undef, nworkers())
    # we prepare the groups of trajectories per worker
    # sync: wait for each worker to finish its task
    @sync begin
        # cycle over workers
        for (w_idx, w) in enumerate(workers())
            # we assign an extra trajectory to the first 'rem' workers
            n_per_worker = traj_per_worker + (w_idx <= rem ? 1 : 0)
            # remotecall assigns a task to a specific worker
            @async begin
                worker_powers_sums = remotecall_fetch(w, ρ_0) do rho # variable redefinition
                    return accumulate_power_evolution([power_evolution(rho) for j in 1:n_per_worker])
                end
                chunk_powers_time[w_idx] = worker_powers_sums[1]
                chunk_powers_thre[w_idx] = worker_powers_sums[2]
            end
        end
    end

    # aggregate the results of this chunk to the progressive sums of the powers
    global sum_power_time_tot .+= sum(chunk_powers_time)
    global sum_power_thre_tot .+= sum(chunk_powers_thre)

    # run time of this chunk and progress printing
    chunk_end_time = time()
    global prog_time += chunk_end_time - chunk_start_time
    println(round(Int64(i * chunk_dim) / NUMBER_OF_TRAJECTORIES * 100, digits = 1), "%. Run time: ", round(prog_time, digits = 2), "s.")
end

# =============================================================================
# 6. AVERAGES COMPUTATION
# =============================================================================

# the accumulated powers are divided by the total number of trajectories
av_power_ev_time = sum_power_time_tot ./ NUMBER_OF_TRAJECTORIES
av_power_ev_thre = sum_power_thre_tot ./ NUMBER_OF_TRAJECTORIES

end_time = time()
println("Total run time: ", round(end_time - start_time, digits = 2), "s.")

# =============================================================================
# 7. RESULTS PRINTING
# =============================================================================

# printing results on files
println("Printing results...")
# --- average power against time ----------------------------------------------
open(processpath * "avepower_" * unravelling * "_against_time.dat", "w") do io
    println(io, "# NTRAJ\t", NUMBER_OF_TRAJECTORIES)
    for (t, pw) in zip(tlist, av_power_ev_time)
        @printf(io, "%.3f\t%.8f\n", t, pw)
    end
end
# --- average power against energy threshold ----------------------------------
open(processpath * "avepower_" * unravelling * "_against_energy_threshold.dat", "w") do io
    println(io, "# NTRAJ\t", NUMBER_OF_TRAJECTORIES)
    for (t, pw) in zip(Elimit, av_power_ev_thre)
        @printf(io, "%.3f\t%.8f\n", t, pw)
    end
end