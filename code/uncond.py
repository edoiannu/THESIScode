# === UNCONDITIONAL EVOLUTION ===

# importing required libraries and objects
import sys
import numpy as np
from my_library.my_system import *
from my_library.my_functions import *
from qutip import mesolve, qeye, sigmam

# variables initialization
inputfile = "input.dat"     # input file
t_f = None                  # evolution final time
dt = None                   # time step
instate = None              # single character variable describing the evolution initial state {p (pure), m (maximally mixed)}
rho0 = None                 # density matrix initial state
alpha_over_kappa = None     # driving field intensity over the system emission rate
eta = None                  # detection efficiency

# passing alpha/kappa and eta from command line
alpha_over_kappa = float(sys.argv[1])
eta = float(sys.argv[2])

# data and results storing directory
if eta == 0.4 and alpha_over_kappa == 1.0:
    directory = "figure1/"
elif eta == 1.0 and alpha_over_kappa == 0.4:
    directory = "figure2/"
else:
    directory = "./"

# reading the remaining input data from file
with open(directory + inputfile, "r") as f:
    # cycle on every line of the file
    for line in f:
        # words separated by a space within a line become the elements of a list
        parts = line.split()
        # it skips empty lines, controls that there are exactly two elements per line (otherwise it skips the line) and skip the comments
        if not line or len(parts) != 2 or line.startswith("#"):
            continue
        key, value = parts
        if key == "INSTATE":
            instate = value 
        elif key == "FINALT":
            t_f = float(value)
        elif key == "dt":
            dt = float(value)

# initial state as density matrix
if instate == "p":
    rho0 = ebasis[0] * ebasis[0].dag()  # ground state
    # for the exited state
    # rho0 = ebasis[1] * ebasis[1].dag()
elif instate == "m":
    rho0 = 0.5 * qeye(DIMH)             # maximally mixed state (one half the identity matrix)
else:
    raise ValueError("The initial state must be pure (p) or maximally mixed (m).")

NUMBER_OF_TIMEINTERVALS = int(t_f / dt)                 # number of time intervals
tlist = np.linspace(0, t_f, NUMBER_OF_TIMEINTERVALS)    # list of time intervals

print("System evolution (initial ", instate, " state, alpha/kappa = ", alpha_over_kappa, ", eta = ", eta, " and ", NUMBER_OF_TIMEINTERVALS, " time intervals)...", sep = "")

# we adoperate the 'mesolve' (master equation solve) method from qutip to solve the unconditional dynamics of the system
unc_evo =  mesolve(HS(alpha_over_kappa),                # system Hamiltonian
                    rho0,                               # initial state
                    tlist,                              # time intervals
                    [sigmam()],                         # list of collapse operators
                    e_ops = [H0],                       # list of operators for which to evaluate the expectation value
                    options = {"store_states": True}    # we store the states corresponding to the times in tlist
                    )

# === Ergotropy and capacity computation ===

print("Ergotropy and capacity computation...")

# list to fill with ergotropy values
erg_results = []
# list to fill with capacity values
cap_results = []

# loop to compute the ergotropy and the capacity at each time
# .states[i] from mesolve gives the state of the i-th time interval
for i in range(NUMBER_OF_TIMEINTERVALS):
    erg_results.append(ergotropy(unc_evo.states[i]))
    cap_results.append(capacity(unc_evo.states[i]))

# saving on a file the results
print("Printing results...")
np.savetxt(directory + "results/en_unc_" + instate + "_alpha" + str(alpha_over_kappa) + ".dat", np.column_stack((tlist, unc_evo.expect[0])), delimiter="\t", fmt=["%.3f", "%.8f"])
np.savetxt(directory + "results/erg_unc_" + instate + "_alpha" + str(alpha_over_kappa) + ".dat", np.column_stack((tlist, erg_results)), delimiter="\t", fmt=["%.3f", "%.8f"])
np.savetxt(directory + "results/cap_unc_" + instate + "_alpha" + str(alpha_over_kappa) + ".dat", np.column_stack((tlist, cap_results)), delimiter="\t", fmt=["%.3f", "%.8f"])