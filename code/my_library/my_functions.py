# === FUNCTIONS DEFINITION ===

# import required libraries and objects
from my_library.my_system import DIMH, ebasis, evalues

# Kronecker delta definition
def delta(i, j):
    if (i == j):
        return 1
    else:
        return 0

# function that computes the ergotropy for a given state rho
def ergotropy(rho):
# evalues and ebasis are the eigenvalues and the eigenstates of the Hamiltonian and they are defined in the preamble
    # the sorting ensures that the first element of the basis corresponds to the most populated state
    # print("State: ", rho)
    rvalues, rstates = rho.eigenstates(sort = 'high')
    # print("Eigenvalues: ", rvalues)
    # print("Eigenstates: ", rstates)
    sum = 0
    for j in range(DIMH):
        for k in range(DIMH):
            # appo = get_scalar(rstates[j].dag() * ebasis[k])   # braket product between <r_j| and |e_k>
            sum += rvalues[j] * evalues[k] * (abs(rstates[j].dag() * ebasis[k])**2  - delta(j, k))
    # print("Ergotropy: ", sum)
    return sum

# function that computes the averaged ergotropy and its variance given a list of different states for each trajectory at a given time
def av_ergotropy(rho):
    sum = 0
    sum2 = 0
    sum3 = 0
    Ntraj = len(rho)    # number of trajectories
    # iteration over the trajectories
    for l in range(Ntraj):
        rho_l = rho[l]
        erg = ergotropy(rho_l)
        sum += erg
        sum2 += erg * erg
        sum3 += erg * erg * erg

    mean = sum / Ntraj
    mean2 = sum2 / Ntraj
    mean3 = sum3 / Ntraj

    return mean, mean2, mean3

# function that computes the capacity for a given state rho
def capacity(rho):
    sum = 0
    rvalues = rho.eigenstates()[0]
    for i in range(DIMH):
        sum += rvalues[i] * (evalues[i] - evalues[DIMH - i - 1])
    return sum

# function that computes the averaged capacity and its variance given a list of different states for each trajectory at a given time
def av_capacity(rho):
    sum = 0
    sum2 = 0
    sum3 = 0
    Ntraj = len(rho)    # number of trajectories
    # iteration over the trajectories
    for l in range(Ntraj):
        rho_l = rho[l]
        cap = capacity(rho_l)
        sum += cap
        sum2 += cap * cap
        sum3 += cap * cap * cap
    
    mean = sum / Ntraj
    mean2 = sum2 / Ntraj
    mean3 = sum3 / Ntraj

    return mean, mean2, mean3