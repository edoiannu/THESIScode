# === DAEMONIC ERGOTROPY AND CAPACITY (AND RESPECTIVE MOMENTA) COMPUTATION ===

# import required libraries and objects
import pickle
import sys
import numpy as np
import os                                           # to count the cores
from concurrent.futures import ProcessPoolExecutor  # for parallel computing  
import time                                         # to count the execution time of a process
from my_library.my_system import *                  # we import everything from my_library
from my_library.my_functions import *
from qutip import qeye
import pickle   # to print complex objects on a file as they are
import gc       # garbage collector for manual memory cleanup

# command that gives the number of usable physical cores (for an eventually parallel compututing)
num_cores = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))

# variables initialization
unravelling = None              # type of unravveling {pd (photodection), phi0 (homodyne detection with phi = 0°), phi90 (homodyne detection with phi = 90°) or hd (heterodyne detection)}
instate = None                  # initial state as a single character variable describing the evolution initial state {p (pure), m (maximally mixed)}
phivalue = None                 # phi value for a homodyne detection               
alpha_over_kappa = None         # driving field intensity over the system emission rate
eta = None                      # detection efficiency
chunk_dim = None                # chunk dimension (for parallel computing)
NUMBER_OF_TRAJECTORIES = None   # number of evolved trajectories
NUMBER_OF_TIMEINTERVALS = None  # number of time intervals per trajectory

# we pass the unravelling as arguments of command line
unravelling = sys.argv[1]

# reading from file data analysis parameters
with open("data_analysis.dat", "r") as f:
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
        elif key == "ALPHA":
            alpha_over_kappa = float(value)
        elif key == "ETA":
            eta = float(value)
        elif key == "CHUNKDIM":
            chunk_dim = int(value)

if unravelling in {"pd", "hod0", "hod90", "hed"}:
    process = unravelling + "_" + instate + "_eta" + str(eta) + "_alpha" + str(alpha_over_kappa)    # process name
else:
    raise ValueError("The unravelling must be photodection, homodyne detection with phi = {0°, 90°} or heterodyne detection {pd, hod0, hod90, hed}")

# initial state as density matrix
if instate == "p":
    rho0 = ebasis[0] * ebasis[0].dag()  # ground state
elif instate == "m":
    rho0 = 0.5 * qeye(DIMH)             # maximally mixed state (one half the identity matrix)
else:
    raise ValueError("The initial state must be pure (p) or maximally mixed (m).")

# directory storing data and results
if eta == 0.4 and alpha_over_kappa == 1.0:
    directory = "figure1/"
elif eta == 1.0 and alpha_over_kappa == 0.4:
    directory = "figure2/"
else:
    directory = "./"

# reading the number of trajectories and time intervals
with open(directory + "/input.dat", "r") as f:
    for line in f:
        parts = line.split()
        if not line or len(parts) != 2 or line.startswith("#"):
            continue
        key, value = parts
        if key == "NTRAJ":
            NUMBER_OF_TRAJECTORIES = int(value)
        elif key == "FINALT":
            t_f = float(value)
        elif key == "dt":
            dt = float(value)

chunk_num = int(NUMBER_OF_TRAJECTORIES / chunk_dim)     # number of chunk
NUMBER_OF_TIMEINTERVALS = int(t_f / dt)                 # number of time intervals
tlist = np.linspace(0, t_f, NUMBER_OF_TIMEINTERVALS)    # list of time intervals

start = time.time()

prog_erg_sum = None # list of the progressive sum and squared sum of the daemonic ergotropies
prog_cap_sum = None # list of the progressive sum and squadre sum of the daemonic capacities

erg_mean = []       # list to fill with the averaged over the trajectories ergotropies at each time
cap_mean = []       # list to fill with the averaged over the trajectories capacities at each time
erg_var = []        # list to fill with the variance of the ergotropies at each time
cap_var = []        # list to fill with the variance of the capacities at each time
erg_skw = []        # list to fill with the skewness of the capacities at each time
cap_skw = []        # list to fill with the skewness of the capacities at each time

print("Averaged quantities computation (unravelling: ", unravelling, ", initial ", instate, " state, alpha/kappa = ", alpha_over_kappa, ", eta = ", eta, ", ", NUMBER_OF_TIMEINTERVALS, " time intervals and ", NUMBER_OF_TRAJECTORIES, " trajectories)...", sep = "")
int_time = 0    # progressive run time

with ProcessPoolExecutor(max_workers = num_cores) as executor:
    for i in range(chunk_num):
        chunk_start_time = time.time()  # we start counting the execution time of the chunk
       # we open the file in binary reading mode
        with open("states/" + process +  "_chunk" + str(i+1) + ".pkl", "rb") as f:
            states = np.array(pickle.load(f), dtype = object)     # we load the i-th chunk, conversion useful to use the function 'executor.map' on the list of states at each time interval
        # rho = np.array(states, dtype = object)
        rho = [states[0 : chunk_dim, j] for j in range(NUMBER_OF_TIMEINTERVALS)] # list of lists of the different trajectories states at given times
        # we map the cores on the time intervals to compute the averaged daemonic ergotropy
        erg_chunk = np.array(list(executor.map(av_ergotropy, rho)))
        cap_chunk = np.array(list(executor.map(av_capacity, rho)))
        del states, rho     # remove reference to solver result to free memory
        gc.collect()        # force garbage collection to release unused memory
        if prog_erg_sum is None:
            prog_erg_sum = erg_chunk
        else:
            prog_erg_sum += erg_chunk
        if prog_cap_sum is None:
            prog_cap_sum = cap_chunk
        else:
            prog_cap_sum += cap_chunk
        chunk_end_time = time.time()    # we end counting the execution time of the chunk
        int_time += chunk_end_time - chunk_start_time
        print(round(int((i + 1) * chunk_dim) / NUMBER_OF_TRAJECTORIES * 100, 1), "%. Run time: ", round(int_time, 2), "s.", sep = "")
erg_mean = prog_erg_sum[:, 0] / chunk_num
cap_mean = prog_cap_sum[:, 0] / chunk_num
erg_var = prog_erg_sum[:, 1] / chunk_num - erg_mean ** 2
cap_var = prog_cap_sum[:, 1] / chunk_num - cap_mean ** 2
# erg_skw = [(prog_erg_sum[:, 2] / chunk_num - 3 * erg_mean * erg_var - erg_mean ** 3) / np.sqrt(i) ** 3 for i in erg_var]
# cap_skw = [(prog_cap_sum[:, 2] / chunk_num - 3 * cap_mean * cap_var - cap_mean ** 3) / np.sqrt(i) ** 3 for i in cap_var]

end = time.time()

# total execution time
print("Total run time: ", round(end - start, 2), "s")

# we print the results on a file
print("Printing results...")
np.savetxt(directory + "results/erg_" + process + ".dat", np.column_stack((tlist, erg_mean)), delimiter="\t", fmt=["%.3f", "%.8f"])
np.savetxt(directory + "results/cap_" + process + ".dat", np.column_stack((tlist, cap_mean)), delimiter="\t", fmt=["%.3f", "%.8f"])
np.savetxt(directory + "results/var_erg_" + process + ".dat", np.column_stack((tlist, erg_var)), delimiter="\t", fmt=["%.3f", "%.8f"])
np.savetxt(directory + "results/var_cap_" + process + ".dat", np.column_stack((tlist, cap_var)), delimiter="\t", fmt=["%.3f", "%.8f"])
# np.savetxt(directory + "results/skw_erg_" + process + ".dat", np.column_stack((tlist, erg_skw)), delimiter="\t", fmt=["%.3f", "%.8f"])
# np.savetxt(directory + "results/skw_cap_" + process + ".dat", np.column_stack((tlist, cap_skw)), delimiter="\t", fmt=["%.3f", "%.8f"])