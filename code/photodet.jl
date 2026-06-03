# === DYNAMICS SIMULATION OF A SYSTEM SUBJECT TO A CONTINOUS PHOTO-DETECTION ===

using Distributed   # for parallel computing

# passing initial state, α/κ and eta from command line
# (Devono essere letti prima di inviarli ai worker tramite @everywhere)
if length(ARGS) != 3
    error("Type simulation parameters separated by a space: initial_state {p (pure state), m (maximally mixed state)} α/κ (driving field intensity over system emission rate) η (detection efficiency).")
end

instate = ARGS[1]
const α_val = parse(Float64, ARGS[2])
const η_val = parse(Float64, ARGS[3])

# Inizializzazione parallela
@everywhere begin
    using Distributed # Parallel environment
    
    # Includiamo le librerie e definiamo le costanti su tutti i core
    include("my_library/my_objects.jl")
    
    const α_over_κ = $α_val
    const η = $η_val
    
    # Spostiamo qui la lettura del file per popolare tlist ovunque
    inputfile = "input.dat"
    t_f = 0.0
    dt = 0.0
    
    for line in eachline(inputfile)
        parts = split(line)
        if isempty(line) || length(parts) != 2 || startswith(line, "#")
            continue
        end
        key, value = parts
        if key == "FINALT"
            global t_f = parse(Float64, value)
        elseif key == "dt"
            global dt = parse(Float64, value)
        end
    end
    
    NUMBER_OF_TIMEINTERVALS = Int64(t_f / dt)
    tlist = range(0, t_f, NUMBER_OF_TIMEINTERVALS + 1)

    # La funzione pevolution ora è visibile globalmente da tutti i worker
    function pevolution(ρ_0)
        ρ_t = ρ_0   # initial state at time t=0
        results = [ρ_t]
        for i in tlist
            ρ_tdt = photodet_kraus(HS(α_over_κ), ρ_t, σ_m, η)
            push!(results, ρ_tdt)
            ρ_t = ρ_tdt
        end
        return results
    end
end

using Printf        # to write on formatted files
using JLD2          # to print trajectories on a file

# Rileggiamo i parametri strutturali sul Master per gestire il ciclo principale
inputfile = "input.dat"
NUMBER_OF_TRAJECTORIES = nothing
chunk_dim = nothing

for line in eachline(inputfile)
    parts = split(line)
    if isempty(line) || length(parts) != 2 || startswith(line, "#")
        continue
    end
    key, value = parts
    if key == "NTRAJ"
        global NUMBER_OF_TRAJECTORIES = parse(Int64, value)
    elseif key == "CHUNKDIM"
        global chunk_dim = parse(Int64, value)
    end
end

# initial state as density matrix (complex in general)
if instate == "p"
    global ρ_0 = ComplexF64[0.00000000 0.00000000 ; 0.00000000 1.00000000]      # ground state
elseif instate == "m"
    global ρ_0 = ComplexF64[0.50000000 0.00000000 ; 0.00000000 0.50000000]  # maximally mixed state
else
    error("The initial state must be pure (p) or maximally mixed (m).")
end

# detection efficiency check
if η_val < 0 || η_val > 1
    error("The detection efficiency must be between 0 and 1.")
end

process = "pd_" * instate * "_eta" * string(η_val) * "_alpha" * string(α_val)    # process name
println("System evolution (initial ", instate, " state, α/κ = ", α_val, ", η = ", η_val, ", ", NUMBER_OF_TIMEINTERVALS, " time intervals and ", NUMBER_OF_TRAJECTORIES, " number of trajectories)...")

prog_time = 0                                                           # progressive run time
start_time = time()                                                     # initial run time
chunk_ind = 0                                                           # chunk index
chunk_num = Int64(NUMBER_OF_TRAJECTORIES / chunk_dim)                   # total number of chunk

for i in 1:chunk_num
    global chunk_ind += 1
    states = []
    chunk_start_time = time()   # we start counting the execution time of the chunk
    
    # Esecuzione parallela distribuita sui worker
    states = pmap(pevolution, [ρ_0 for j in 1:chunk_dim])
    
    @save "states/" * process * "_chunk" * string(chunk_ind) * ".jld2" states 
    chunk_end_time = time()     # we end counting the execution time of the chunk
    global prog_time += chunk_end_time - chunk_start_time
    println(round(Int64(i * chunk_dim) / NUMBER_OF_TRAJECTORIES * 100, digits = 1), "%. Run time: ", round(prog_time, digits = 2), "s.")
end

end_time = time()   # final run time
println("Total run time: ", round(end_time - start_time, digits = 2), "s.")
