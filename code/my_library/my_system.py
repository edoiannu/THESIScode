# === SYSTEM DEFINITION ===

# import required libraries and objects
from qutip import sigmax, sigmaz, basis, expect

# we have redefined the system coefficients and the collapse operator:
# alpha -> alpha / kappa
# t -> kappa * t
# sqrt(k) c -> c

# Hilbert space dimension
DIMH = 2

# system Hamiltonian
H0 = 0.5 * (sigmaz() + 1)
# we set omega_0 = 1 since we are going to plot the results per unit energy

# Hamiltonian which describes the system evolution
def HS(alpha_over_kappa):
    return alpha_over_kappa * sigmax()

# energy eigenstates
ebasis = [basis(DIMH,                               # number of basis states, for a single qubit they are two: |0> & |1>
                i) for i in reversed(range(DIMH))   # the function 'reversed' unsures the eigenstate corresponding to the lowest energy (ground state) is the first element of the basis
                ]

# energy eigenvalues
evalues = [expect(H0, ebasis[i]) for i in range(DIMH)]
