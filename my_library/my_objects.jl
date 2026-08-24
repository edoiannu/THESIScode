# === OBJECT DEFINITIONS ===
# we have redefined the system coefficients and the collapse operator:
# alpha -> alpha / kappa
# t -> kappa * t
# sqrt(k) c -> c

# importing required libraries
using LinearAlgebra

DIMH = 2    # Hilbert space dimension

# Pauli matrices
σ_x = [0 1 ; 1 0]
σ_y = [0 -1im ; 1im 0]
σ_z = [1 0 ; 0 -1]
σ_m = (1 / 2) * (σ_x - 1im * σ_y)
σ_p = (1 / 2) * (σ_x + 1im * σ_y)

# Kronecker delta definition
function delta(i, j)
    if i == j
        return 1
    else
        return 0
    end
end

# function that returns che Hermitian conjugate of a generic object O
function dagger(O)
    return transpose(conj(O))
end

# function that return the commutator between two generic objects A and B
function commutator(A, B)
    return A * B - B * A
end

# function that returns the anticommutator between two generic objects A and B
function anticommutator(A, B)
    return A * B + B * A
end

# function that returns the Lindbladian superoperator given a collapse operator c and a density matrix ρ
function D(c, ρ)
    return c * ρ * dagger(c) - 0.5 * anticommutator(dagger(c) * c, ρ)
end

# system Hamiltonian
H0 = 0.5 * (σ_z + [1 0 ; 0 1])
# we set ω_0 = 1 since we are going to plot the results per unit energy

E = eigen(H0)
# energy eigenvalues
evalues = E.values
# energy eigenstates
ebasis = E.vectors

# Hamiltonian which describes the system evolution
function HS(α_over_κ)
    return α_over_κ * σ_x
end

# function that rules the unconditional evolution of a system given:
# - the system Hamiltonian HS
# - the initial state (in the form of density matrix ρ_0)
# - the list of time intervals tlist
# - the collapse operator c
function uncond_evo(HS, ρ_0, tlist, c)
    ρ_t = ρ_0                   # state at time t
    ρ_tdt = nothing             # evolved state at time t+dt
    results = [ρ_t]             # list to fill with the state evolution
    dt = tlist[2] - tlist[1]    # time interval
    # system evolution
    for i in tlist
        ρ_tdt = ρ_t - 1im * commutator(ρ_t, HS) * dt + D(c, ρ_t) * dt   # master equation
        push!(results, ρ_tdt)                                           # updating the state evolution
        ρ_t = ρ_tdt                                                     # evolved state at succesive time t+dt
    end
    # returning the state history
    return results
end

# function that returns the state evolution of a system subject to a continuous photo-detection given:
# - the system Hamiltonian HS
# - the initial state (in the form of density matrix ρ_0)
# - the list of time intervals tlist
# - the collapse operator c
# - the detection efficiency η
function photodet_kraus(HS, ρ_t, c, η)
    # ρ_t = ρ_0                   # initial state at time t=0
    num = nothing               # numerator of the state 
    ρ_tdt = nothing             # evolved stated
    # results = [ρ_t]
    # kraus operators
    kraus = nothing
    M0 = I - 1im * HS * dt - 0.5 * dagger(c) * c * dt   # no jump operator
    M1 = sqrt(η) * c * sqrt(dt)                         # jump operator
    # conditional jump probability
    pjump = nothing
    function P(ρ)
        return real(η * tr(dagger(c) * c * ρ) * dt)
    end
    r = nothing  # random number to decide if a jump occurs or not
    # for i in tlist
    pjump = P(ρ_t)      # jump probability
    r = rand()          # random number
    if r < pjump        # if a jump occurs
        kraus = M1
    else                # if not
        kraus = M0
    end
    num = kraus * ρ_t * dagger(kraus) + (1 - η) * c * ρ_t * dagger(c) * dt  # numerator of the state evolution
    ρ_tdt = num / tr(num)    # state evolution
    # push!(results, ρ_t)    # updating the state evolution
    # end
    return ρ_tdt
end

# function that returns the state evolution of a system subject to a continuous homodyne-detection given:
# - the system Hamiltonian HS
# - the initial state (in the form of density matrix ρ_0)
# - the list of time intervals tlist
# - the collapse operator c
# - the detection efficiency η
# - the detection angle ϕ
function dyne_kraus(HS, ρ_t, cops, η, heterodyne)
    num = nothing
    ρ_tdt = nothing             # evolved states
    dy_t = nothing
    dy1_t = nothing
    dy2_t = nothing
    M_dy = nothing
    # photo-current
    if heterodyne == true
        dy1_t = sqrt(η/2) * tr((cops + dagger(cops)) * ρ_t) * dt + sqrt(dt) * randn()
        dy2_t = sqrt(η/2) * tr(1im * (cops - dagger(cops)) * ρ_t) * dt + sqrt(dt) * randn()
        M_dy = I - 1im * HS * dt - 0.5 * dagger(cops) * cops * dt + sqrt(η/2) * cops * dy1_t + 1im * sqrt(η/2) * cops * dy2_t
    else
        # M_dy = M(ρ_t, cops)
        dy_t = sqrt(η) * tr((cops + dagger(cops)) * ρ_t) * dt + sqrt(dt) * randn()
        M_dy = I - 1im * HS * dt - 0.5 * dagger(cops) * cops * dt + sqrt(η) * cops * dy_t
    end
    num = M_dy * ρ_t * dagger(M_dy) + (1 - η) * cops * ρ_t * dagger(cops) * dt
    ρ_tdt = num / tr(num)
    return ρ_tdt
end

# function that returns the mean energy of the system with Hamiltonian H0 given its state ρ
function energy(ρ)
    # expectation value: trace of state * operator
    return tr(ρ * H0)
end

# function that computes the ergotropy for a given state ρ
function ergotropy(ρ)
    # using "eigen" from "LinearAlgebra"
    # with "sortby" we invert the eigenvalues x order (decreasing order), inverting their signs
    # in general we have complex eigenvalues
    F = eigen(ρ, sortby = x -> -abs(x))
    total = 0
    rvalues = F.values
    rstates = F.vectors
    for j in 1:DIMH
        for k in 1:DIMH
            total += rvalues[j] * evalues[k] * (abs(dagger(rstates[j]) * dagger(ebasis[k]))^2 - delta(j, k))
        end
    end

    return real(total)
end

# function that computes the averaged ergotropy and its variance given a list of different states for each trajectory at a given time
function av_ergotropy(ρ)
    sum = 0
    sum2 = 0
    sum3 = 0
    Ntraj = length(ρ)   # nummber of trajectories
    # iterations over the trajectories
    for l in 1:Ntraj
        ρ_l = ρ[l]
        erg = ergotropy(ρ_l)
        sum += erg
        sum2 += erg * erg
        sum3 += erg * erg * erg
    end

    mean = sum / Ntraj
    mean2 = sum2 / Ntraj
    mean3 = sum3 / Ntraj

    return [mean, mean2, mean3]
end

# function that computes the capacity for a given state ρ
function capacity(ρ)
    F = eigen(ρ)
    total = 0
    rvalues = F.values
    for i in 1:DIMH
        total += rvalues[i] * (evalues[i] - evalues[DIMH - i + 1])
    end
    return real(total)
end

# function that computes the averaged capacity and its variance given a list of different states for each trajectory at a given time
function av_capacity(ρ)
    sum = 0
    sum2 = 0
    sum3 = 0
    Ntraj = length(ρ)   # nummber of trajectories
    # iterations over the trajectories
    for l in 1:Ntraj
        ρ_l = ρ[l]
        erg = capacity(ρ_l)
        sum += erg
        sum2 += erg * erg
        sum3 += erg * erg * erg
    end

    mean = sum / Ntraj
    mean2 = sum2 / Ntraj
    mean3 = sum3 / Ntraj

    return [mean, mean2, mean3]
end