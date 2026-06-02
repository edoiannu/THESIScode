# === DAEMONIC ERGOTROPY AND CAPACITY (AND RESPECTIVE MOMENTA) COMPUTATION FROM STATE DYNAMICS ===

# import required libraries and objects
include("my_library/my_objects.jl")
using Printf        # to write on formatted files
using Distributed   # for parallel computing
using JLD2          # to print trajectories on a file

# variables initialization
inputfile = "data_analysis.dat"     # input file
unravelling = nothing               # type of unravveling {pd (photodection), phi0 (homodyne detection with phi = 0°), phi90 (homodyne detection with phi = 90°) or hd (heterodyne detection)}
instate = nothing                   # initial state as a single character variable describing the evolution initial state {p (pure), m (maximally mixed)}
ϕ = nothing                 
α_over_κ = nothing                  # driving field intensity over the system emission rate
η = nothing                         # detection efficiency
chunk_dim = nothing                 # chunk dimension (for parallel computing)
NUMBER_OF_TRAJECTORIES = nothing    # number of evolved trajectories
NUMBER_OF_TIMEINTERVALS = nothing   # number of time intervals per trajectory

println("=== DAEMONIC ERGOTROPY AND CAPACITY (AND RESPECTIVE MOMENTA) COMPUTATION FROM STATES DYNAMICS ===")
unravelling = ARGS[1]

if length(ARGS) != 1
    error("Type considered unravelling {pd, hod[detection_angle], hed}")
end

# reading from file data analysis parameters
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
    elseif key == "CHUNKDIM"
        global chunk_dim = parse(Int64, value)
    end
end

process = unravelling * "_" * instate * "_eta" * string(η) * "_alpha" * string(α_over_κ)  # process name

# reading the number of trajectories and time intervals
for line in eachline("input.dat")
    parts = split(line)
    if isempty(line) || length(parts) != 2 || startswith(line, "#")
        continue
    end
    key, value = parts
    if key == "FINALT"
        global t_f = parse(Float64, value)
    elseif key == "dt"
        global dt = parse(Float64, value)
    elseif key == "NTRAJ"
        global NUMBER_OF_TRAJECTORIES = parse(Int64, value)
    end
end

NUMBER_OF_TIMEINTERVALS = Int64(t_f / dt)           # number of time intervals
tlist = range(0, t_f, NUMBER_OF_TIMEINTERVALS + 1)  # list of time intervals ("+ 1" because it starts with t=0)

prog_erg_sum = nothing  # list of the progressive sum and squared sum of the daemonic ergotropies
prog_cap_sum = nothing  # list of the progressive sum and squadre sum of the daemonic capacities
erg_mean = []           # list to fill with the averaged over the trajectories ergotropies at each time
cap_mean = []           # list to fill with the averaged over the trajectories capacities at each time
erg_var = []            # list to fill with the variance of the ergotropies at each time
cap_var = []            # list to fill with the variance of the capacities at each time
# erg_skw = []            # list to fill with the skewness of the capacities at each time
# cap_skw = []            # list to fill with the skewness of the capacities at each time

println("Averaged quantities computation (unravelling: ", unravelling, ", initial ", instate, " state, α/κ = ", α_over_κ, ", η = ", η, ", ", NUMBER_OF_TIMEINTERVALS, " time intervals and ", NUMBER_OF_TRAJECTORIES, " trajectories)...")

prog_time = 0                                           # progressive run time
start_time = time()                                     # total run time
chunk_num = Int64(NUMBER_OF_TRAJECTORIES / chunk_dim)   # total number of chunk

for i in 1:chunk_num
    states = []
    ρ = []
    chunk_start_time = time()  # we start counting the execution time of the chunk
    @load "states/" * process * "_chunk" * string(i) * ".jld2" states
    for j in 1:NUMBER_OF_TIMEINTERVALS
        push!(ρ, [states[k][j] for k in 1:chunk_dim])
    end
    # we map the cores on the time intervals to compute the averaged daemonic ergotropy
    erg_chunk = pmap(av_ergotropy, ρ)
    cap_chunk = pmap(av_capacity, ρ)
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
    chunk_end_time = time()     # we end counting the execution time of the chunk
    global prog_time += chunk_end_time - chunk_start_time
    println(round(Int64(i * chunk_dim) / NUMBER_OF_TRAJECTORIES * 100, digits = 1), "%. Run time: ", round(prog_time, digits = 2), "s.")
end

erg_mean = [x[1] for x in prog_erg_sum] ./ chunk_num
cap_mean = [x[1] for x in prog_cap_sum] ./ chunk_num
erg_var = [x[2] for x in prog_erg_sum] ./ chunk_num - erg_mean .^ 2
cap_var = [x[2] for x in prog_cap_sum] ./ chunk_num - cap_mean .^ 2

end_time = time()   # final run time

# total execution time
println("Total run time: ", round(end_time - start_time, digits = 2), "s")

# we print the results on a file
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