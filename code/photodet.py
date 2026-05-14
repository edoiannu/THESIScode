# === DYNAMICS OF A SYSTEM SUBJECT TO A CONTINOUS PHOTO-DETECTION ===

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
import pickle   # to print complex objects on a file as they are
import gc       # garbage collector for manual memory cleanup

# command that gives the number of usable physical cores (for an eventual parallel compututing)
num_cores = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))

# variables initialization
inputfile = "input.dat"         # input file
t_f = None                      # evolution final time
dt = None                       # time step 
NUMBER_OF_TRAJECTORIES = None   # number of trajectories to evolve
instate = None                  # initial state as a single character variable describing the evolution initial state {p (pure), m (maximally mixed)}
rho0 = None                     # initial state as a density matrix
alpha_over_kappa = None         # driving field intensity over the system emission rate
eta = None                      # detection efficiency
chunk_dim = None                # chunk dimension (number of trajectories to evolve simultaneously)

# we pass alpha/kappa and eta as arguments of command line
alpha_over_kappa = float(sys.argv[1])
eta = float(sys.argv[2])

# directory storing data and results
if eta == 0.4 and alpha_over_kappa == 1.0:
    directory = "figure1/"
elif eta == 1.0 and alpha_over_kappa == 0.4:
    directory = "figure2/"
else:
    directory = "./"

# reading from file simulation parameters
with open(directory + inputfile, "r") as f:
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
        elif key == "NTRAJ":
            NUMBER_OF_TRAJECTORIES = int(value)
        elif key == "CHUNKDIM":
            chunk_dim = int(value)

# initial state as density matrix
if instate == "p":
    rho0 = ebasis[0] * ebasis[0].dag()  # ground state
elif instate == "m":
    rho0 = 0.5 * qeye(DIMH)             # maximally mixed state (one half the identity matrix)
else:
    raise ValueError("The initial state must be pure (p) or maximally mixed (m).")

process = "pd_" + instate + "_eta" + str(eta) + "_alpha" + str(alpha_over_kappa)    # process name
NUMBER_OF_TIMEINTERVALS = int(t_f / dt)                     # number of time intervals
tlist = np.linspace(0, t_f, NUMBER_OF_TIMEINTERVALS + 1)    # list of time intervals

print("System evolution (initial ", instate, " state, alpha/kappa = ", alpha_over_kappa, ", eta = ", eta, ", ", NUMBER_OF_TIMEINTERVALS, " time intervals and ", NUMBER_OF_TRAJECTORIES, " trajectories)...", sep = "")

# system operators
c = sigmam()    # collapse operator
# kraus operators
M0 = qeye(DIMH) - 1j * HS(alpha_over_kappa) * dt - 0.5 * c.dag() * c * dt   # no jump
M1 = np.sqrt(eta) * c * np.sqrt(dt)                                         # jump
# conditional jump probability
def P(rhot):
    return (eta * (c.dag() * c * rhot).tr() * dt).real
# function that returns the state evolution of a system subject to a contiuous photo-detection
def smesolve_kraus(rho_start):
    
    rhot = rho_start    # initial state
    results = [rhot]
    # system evolution
    for i in range(1, NUMBER_OF_TIMEINTERVALS):
        pjump = P(rhot)     # jump probability
        r = rnd.random()
        kraus = None        # kraus operator
        if r <= pjump: kraus = M1   # if a jump occurs
        else: kraus = M0            # if not
        num = kraus * rhot * kraus.dag() + (1 - eta) * c * rhot * c.dag() * dt
        rhot = num / num.tr()       # successive conditional state
        results.append(rhot)
    
    return results

print("System evolution (initial ", instate, " state, alpha/kappa = ", alpha_over_kappa, ", eta = ", eta, ", ", NUMBER_OF_TIMEINTERVALS, " time intervals and ", NUMBER_OF_TRAJECTORIES, " trajectories)...", sep = "")

int_time = 0    # progressive run time
start = time.time()
chunk_num = 0   # chunk number
with ProcessPoolExecutor(max_workers = num_cores) as executor:
    for i in range(0, NUMBER_OF_TRAJECTORIES, chunk_dim):
        chunk_num += 1
        states = []
        chunk_start_time = time.time()  # we start counting the execution time of the chunk
        # we map in parallel the stochastic Kraus master equation solver at chunk of initial states with dimension chunk_dim
        states = list(executor.map(smesolve_kraus, [rho0] * (chunk_dim)))
        # we append the states on a file
        with open("states/" + process + "_chunk" + str(chunk_num) + ".pkl", "ab") as f:
            pickle.dump(states, f)      # we print on a file the states evolutions of each chunk
        del states          # remove reference to solver result to free memory
        gc.collect()        # force garbage collection to release unused memory
        chunk_end_time = time.time()    # we end counting the execution time of the chunk
        int_time += chunk_end_time - chunk_start_time
        print(round(int(i + chunk_dim) / NUMBER_OF_TRAJECTORIES * 100, 1), "%. Run time: ", round(int_time, 2), "s.", sep = "")

end = time.time()

# total execution time
print("Total run time: ", round(end - start, 2), "s", sep = "")