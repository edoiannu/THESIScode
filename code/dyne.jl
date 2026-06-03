# === DYNAMICS SIMULATION OF A SYSTEM SUBJECT TO A CONTINOUS GENERIC DYNE DETECTION ===

using Distributed   # for parallel computing

# Controllo preliminare sulla lunghezza minima degli argomenti
if length(ARGS) < 4 || length(ARGS) > 5
    error("Type simulation parameters separated by a space:\n- For homodyne: 'hod' ϕ instate α/κ η\n- For heterodyne: 'hed' instate α/κ η")
end

# Lettura asimmetrica degli argomenti da linea di comando sul Master
const det_type = ARGS[1]
if det_type == "hod"
    const ϕ_val = parse(Int64, ARGS[2])
    instate = ARGS[3]
    const α_val = parse(Float64, ARGS[4])
    const η_val = parse(Float64, ARGS[5])
    const het_val = false
elseif det_type == "hed"
    const ϕ_val = 0 # Valore di fallback per heterodyne
    instate = ARGS[2]
    const α_val = parse(Float64, ARGS[3])
    const η_val = parse(Float64, ARGS[4])
    const het_val = true
else
    error("Detection type must be homodyne ('hod') or heterodyne ('hed').")
end

# Inizializzazione parallela dei worker e delle dipendenze
@everywhere begin
    using Distributed # Parallel environment
    
    # Inclusione oggetti e librerie locali su tutti i core
    include("my_library/my_objects.jl")
    
    # Iniezione delle costanti lette dal Master
    const α_over_κ = $α_val
    const η = $η_val
    const heterodyne = $het_val
    const ϕ = $ϕ_val
    const c = σ_m
    
    # Costruzione dell'operatore di collasso specifico su ogni worker
    clean(x; tol = 1e-14) = abs(x) < tol ? 0 : x
    if $det_type == "hod"
        const cops = (clean(cos(deg2rad(ϕ))) + 1im * sin(deg2rad(ϕ))) * c
    else
        const cops = c
    end
    
    # Lettura dei parametri temporali del file input.dat su tutti i core
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

    # La funzione pevolution ora ha pieno accesso a tutte le costanti di calcolo, versione ottimizzata senza push! dinamici
    function pevolution(ρ_0)
        # Pre-allochiamo lo spazio esatto per tutti gli intervalli di tempo
        len_tlist = length(tlist)
        results = Vector{Matrix{ComplexF64}}(undef, len_tlist)
        
        results[1] = ρ_0   # initial state at time t=0
        
        # Usiamo un indice per evitare l'allocazione di memoria dinamica
        idx = 1
        for i in 1:(len_tlist - 1)
            # Supponiamo che dyne_kraus accetti l'operatore AND se necessario: flag & 0x01
            ρ_tdt = dyne_kraus(HS(α_over_κ), results[idx], cops, η, heterodyne)
            idx += 1
            results[idx] = ρ_tdt
        end
        return results
    end
end

using Printf        # to write on formatted files
using JLD2          # to print trajectories on a file

# Lettura delle restanti variabili globali strutturali sul Master
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

# Generazione della stringa di processo univoca
if det_type == "hod"
    process = "hod" * string(ϕ_val) * "_" * instate * "_eta" * string(η_val) * "_alpha" * string(α_val)  
else
    process = "hed_" * instate * "_eta" * string(η_val) * "_alpha" * string(α_val)
end

# Inizializzazione della matrice di densità iniziale (ρ_0)
if instate == "p"
    global ρ_0 = ComplexF64[0.00000000 0.00000000 ; 0.00000000 1.00000000]      # ground state
elseif instate == "m"
    global ρ_0 = ComplexF64[0.50000000 0.00000000 ; 0.00000000 0.50000000]  # maximally mixed state
else
    error("The initial state must be pure (p) or maximally mixed (m).")
end

if η_val < 0 || η_val > 1
    error("The detection efficiency must be between 0 and 1.")
end

println("System evolution (initial ", instate, " state, α/κ = ", α_val, ", η = ", η_val, ", ", NUMBER_OF_TIMEINTERVALS, " time intervals and ", NUMBER_OF_TRAJECTORIES, " number of trajectories)...")

# ... (tutta la parte iniziale e l'inizializzazione rimangono identiche) ...

start_time = time()
chunk_ind = 0
chunk_num = Int64(NUMBER_OF_TRAJECTORIES / chunk_dim)

# we compute how many trajectories within a chunck are up to each worker
traiettorie_per_worker = ceil(Int64, chunk_dim / nworkers())

for i in 1:chunk_num
    global chunk_ind += 1
    chunk_start_time = time()
    
    # we prepare the groups of trajectories per worker
    # sync: wait for each worker to finisci its task
    @sync begin
        # cycle over workers
        for (w_idx, w) in enumerate(workers())
            # remotecall asigns a task to a specific worker and performs a variable redefinition
            @async remotecall_fetch(w, ρ_0, chunk_ind, w_idx) do rho, c_id, w_id
                
                # Il worker esegue il suo sotto-chunk localmente
                local_states = [pevolution(rho) for j in 1:traiettorie_per_worker]
                
                # CRUCIALE: Ogni worker salva direttamente il SUO file sul disco.
                # Il Master non deve fare nulla, non si formano imbuti!
                filename = "states/$(process)_chunk$(c_id)_worker$(w_id).jld2"
                @save filename local_states
            end
        end
    end
    
    chunk_end_time = time()
    println("Chunk ", chunk_ind, "/", chunk_num, " completato in ", round(chunk_end_time - chunk_start_time, digits=2), "s.")
    flush(stdout)
end

end_time = time()
println("Total run time: ", round(end_time - start_time, digits = 2), "s.")
