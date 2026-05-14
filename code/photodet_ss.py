# import required libraries and objects
import sys
import numpy as np
import random as rnd                                # to deal with pseudorandom numbers
import os                                           # to count the cores
from concurrent.futures import ProcessPoolExecutor  # for parallel computing  
import time                                         # to count the execution time of a process
from my_library.my_system import *                  # we import everything from my_library
from my_library.my_functions import *
from qutip import sigmam, qeye

# command that gives the number of usable physical cores
num_cores = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))

# === STEADY STATES DYNAMICS OF A SYSTEM SUBJECT TO A CONTINOUS PHOTO-DETECTION ===

# variables initialization
t_f = None
dt = None
alpha_f = None                      # final value of alpha over kappa
NUMBER_OF_ALPHAPOINTS = None        # number of alpha over kappa points
NUMBER_OF_TRAJECTORIES = None
instate = None
eta = None
chunk_dim = None    # chunk dimension

# we pass eta as argument of command line
eta = float(sys.argv[1])

# reading from file simulation parameters
with open("figure3/input.dat", "r") as f:
    # cycle on every line of the file
    for line in f:
        # words separated by a space within a line become the elements of a list
        parts = line.split()
        # it skips empty lines, controls that there are exactly two elements per line (otherwise it skips the line) and skips the comments
        if not line or len(parts) != 2 or line.startswith("#"):
            continue
        key, value = parts
        if key == "INSTATE":
            instate = value  
        elif key == "FINALT":
            t_f = float(value)
        elif key == "dt":
            dt = float(value)
        elif key == "FINALALPHA":
            alpha_f = float(value) 
        elif key == "ALPHAPOINTS":
            NUMBER_OF_ALPHAPOINTS = int(value)
        elif key == "NTRAJ":
            NUMBER_OF_TRAJECTORIES = int(value)
        elif key == "CHUNKDIM":
            chunk_dim = int(value)

process = "pd_eta" + str(eta)                               # process name
d_alpha = float(alpha_f / NUMBER_OF_ALPHAPOINTS)            # alpha over kappa interval width
NUMBER_OF_TIMEINTERVALS = int(t_f / dt)                     # number of time intervals
tlist = np.linspace(0, t_f, NUMBER_OF_TIMEINTERVALS)        # list of time intervals
alphalist = np.linspace(0, alpha_f, NUMBER_OF_ALPHAPOINTS)  # list of alpha value

# system operators
c = sigmam()    # collapse operator
# kraus operators
def M0(alpha_over_kappa): return qeye(DIMH) - 1j * HS(alpha_over_kappa) * dt - 0.5 * c.dag() * c * dt   # no jump
M1 = np.sqrt(eta) * c * np.sqrt(dt)                                                                     # jump
# conditional jump probability
def P(rhot):
    return (eta * (c.dag() * c * rhot).tr() * dt).real
# function that returns the staedy-state of a system subject to a contiuous photo-detection
def ss_smesolve_kraus(alpha_over_kappa):
    # initial state as density matrix
    rho0 = None
    if instate == "p":
        rho0 = ebasis[0] * ebasis[0].dag()  # ground state
    elif instate == "m":
        rho0 = 0.5 * qeye(DIMH)             # maximally mixed state (one half the identity matrix)
    else:
        rho0 = ebasis[0] * ebasis[0].dag()  # initial state set as pure by default

    rhot = rho0
    # system evolution
    for i in range(1, NUMBER_OF_TIMEINTERVALS):
        pjump = P(rhot)     # jump probability
        r = rnd.random()
        kraus = None        # kraus operator
        if r <= pjump: kraus = M1           # if a jump occurs
        else: kraus = M0(alpha_over_kappa)  # if not
        num = kraus * rhot * kraus.dag() + (1 - eta) * c * rhot * c.dag() * dt
        rhot = num / num.tr()       # successive conditional state

    return rhot

ss_ergotropies = []     # steady states daemonic ergotropies
ss_capacities = []      # steady states daemonic capacities

print("Steady states evolutions (eta = ", eta, ", ", NUMBER_OF_ALPHAPOINTS, " alpha/kappa points, ", NUMBER_OF_TIMEINTERVALS, " time intervals and ", NUMBER_OF_TRAJECTORIES, " trajectories)...", sep = "")

int_time = 0    # progressive run time
start = time.time()
i = 0
with ProcessPoolExecutor(max_workers = num_cores) as executor:
    for a in alphalist:
        i += 1
        # states = []
        chunk_start_time = time.time()  # we start counting the execution time of the chunk
        # we map in parallel the stochastic Kraus master equation solver at chunk of initial states with dimension chunk_dim
        steadystates = list(executor.map(ss_smesolve_kraus, [a] * (NUMBER_OF_TRAJECTORIES)))
        ss_ergotropies.append(av_ergotropy(steadystates)[0])
        ss_capacities.append(av_capacity(steadystates)[0])
        chunk_end_time = time.time()    # we end counting the execution time of the chunk
        int_time += chunk_end_time - chunk_start_time
        print("alpha/kappa = ", round(a, 5), ", ", round(int(i) / NUMBER_OF_ALPHAPOINTS * 100, 1), "%. Run time: ", round(int_time, 2), "s.",  sep = "")

end = time.time()

# we print the results on a file
np.savetxt("figure3/results/ss_erg_" + process + ".dat", np.column_stack((alphalist, ss_ergotropies)), delimiter="\t", fmt=["%.3f", "%.8f"])
np.savetxt("figure3/results/ss_cap_" + process + ".dat", np.column_stack((alphalist, ss_capacities)), delimiter="\t", fmt=["%.3f", "%.8f"])

