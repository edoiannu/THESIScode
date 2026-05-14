# import required libraries and objects
import sys
import numpy as np
import os                                           # to count the cores
import time                                         # to count the execution time of a process
from my_library.my_system import *                  # we import everything from my_library
from my_library.my_functions import *
from qutip import sigmam, qeye, smesolve
import pickle   # to print complex objects on a file as they are
import gc       # garbage collector for manual memory cleanup

# command that gives the number of usable physical cores (for an eventual parallel compututing)
num_cores = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))

# === DYNAMICS OF A SYSTEM SUBJECT TO A CONTINOUS HOMODYNE DETECTION ===

# variables initialization
t_f = None
dt = None
NUMBER_OF_TRAJECTORIES = None
instate = None
rho0 = None
alpha_over_kappa = None
eta = None
cops = None # collapse operator depending on the value of phi

# we pass phi, alpha/kappa and eta as arguments of command line
phi = int(sys.argv[1])
alpha_over_kappa = float(sys.argv[2])
eta = float(sys.argv[3])

if phi == 0:
    cops = sigmam()
elif phi == 90:
    cops = 1j * sigmam()
else:
    raise ValueError("phi must be 0° or 90°.")

# directory storing data and results
if eta == 0.4 and alpha_over_kappa == 1.0:
    directory = "figure1/"
elif eta == 1.0 and alpha_over_kappa == 0.4:
    directory = "figure2/"
else:
    directory = "./"

with open(directory + "input.dat", "r") as f:
    # cycle on every line of the file
    for line in f:
        # words separated by a space within a line become different elements of a list
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

process = "hod" + str(phi) + "_" + instate + "_eta" + str(eta) + "_alpha" + str(alpha_over_kappa)    # process name
NUMBER_OF_TIMEINTERVALS = int(t_f / dt)                 # number of time intervals
tlist = np.linspace(0, t_f, NUMBER_OF_TIMEINTERVALS)    # list of time intervals

print("System evolution (initial ", instate, " state, alpha/kappa = ", alpha_over_kappa, ", eta = ", eta, ", ", NUMBER_OF_TIMEINTERVALS, " time intervals and ", NUMBER_OF_TRAJECTORIES, " trajectories)...", sep = "")

int_time = 0    # progressive run time
start = time.time()
chunk_num = 0   # for chunk counting
for i in range(0, NUMBER_OF_TRAJECTORIES, chunk_dim):
    chunk_num += 1
    chunk_start_time = time.time()  # we start counting the execution time of the chunk
    hod = smesolve(
        HS(alpha_over_kappa),
        rho0,
        tlist,
        c_ops = [np.sqrt(1 - eta) * cops],
        sc_ops = [np.sqrt(eta) * cops],
        ntraj = chunk_dim,
        options = {"store_states": True,
                "keep_runs_results": True,
                "map": "parallel",          # parallel computing
                "num_cpus": num_cores,      # number of cpus to use when running in parallel
                "progress_bar": False
                }
    )
    with open("states/" + process + "_chunk" + str(chunk_num) + ".pkl", "ab") as f:
        pickle.dump(hod.states, f)  # we print on a file the states evolutions of each chunk
    del hod              # remove reference to solver result to free memory
    gc.collect()         # force garbage collection to release unused memory
    chunk_end_time = time.time()    # we end counting the execution time of the chunk
    int_time += chunk_end_time - chunk_start_time
    print(round(int(i + chunk_dim) / NUMBER_OF_TRAJECTORIES * 100, 1), "%. Run time: ", round(int_time, 2), "s.", sep = "")

end = time.time()

# total execution time
print("Total run time: ", round(end - start, 2), "s", sep = "")