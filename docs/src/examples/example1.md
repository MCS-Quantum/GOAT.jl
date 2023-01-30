# Optimizing a gaussian pulse for a single qubit

This is perhaps the simplest example of a quantum optimal control problem. The physical task is to invert a qubit's population from he ground state $\ket{0}$ to the orthogonal state $\ket{1}$.

Given a qubit Hamiltonian 

```math
H(t) = \frac{\omega}{2}\sigma_z + \Omega(t)\sigma_x
```

one can determine analytically that there are a family of pulses which will accomplish the control task if the following criteria is met:

```math
\int_0^t d\tau \Omega(\tau) = \pi
```

one common choice of $\Omega(t)$ is a Gaussian function (due to an array of reasons outside the scope of this example). 

For this example we will fix the mean and standard deviation of the Gaussian to $\mu=T/2$ and $\sigma = T/8$. Therefore, we must determine the amplitude of the Gaussian in order to satisfy the topological criteraia and create population inversion.

## Imports

```@example test1
using GOAT
using LinearAlgebra, SparseArrays # For instantiating necessary matrices
using DifferentialEquations # Load differetial equation algorithms
using Optim, LineSearches # Load optimization and linesearch algorithms
```

## Defining the `ControllableSystem`

We begin by defining the necessary operators for the Hamiltonian

```@example test1
σx = sparse(ComplexF64[0 1 ; 1  0])# The Pauli X matrix
σy = sparse(ComplexF64[0 -im ; im 0]) # The Pauli Y matrix
σz = sparse(ComplexF64[1 0 ; 0 -1]) # The Pauli Z matrix
σI = sparse(ComplexF64[1 0 ; 0  1]) # The identity matrix
```

Next we define the Hamiltonian parameters and the control basis operators

```@example test1
ω = 5*2pi # Setting a qubit frequency
T = 10.0 # in arbitrary time units
H0 = (ω/2)*σz # Setting the drift component of the Hamiltonian
basis_ops = [σx] # The control basis operator
```

Next we choose a control ansatz specifying only a single basis function

```@example test1
N_basis_funcs = 1
Ω(p,t,i) = gaussian_coefficient(p,t,i,N_basis_funcs)*cos(ω*t) # The control ansatz is a gaussian envelope of a cosine wave at frequency ω
∂Ω(p,t,i,l) = ∂gaussian_coefficient(p,t,i,l,N_basis_funcs)*cos(ω*t) # The derivative of the control ansatz
```

and specify a rotating frame

```@example test1
rotating_frame_generator = -Diagonal(Matrix(H0)) # Moving into the interaction picture
```

Finally we are able to specify our `ControllableSystem`:

```@example test1
sys = ControllableSystem(H0, basis_ops, rotating_frame_generator, Ω, ∂Ω)
```

## Defining the `QOCProblem`

First we define our target unitary operator

```@example test1
U_target = deepcopy(σx) # The targt unitary: X|0> = |1>
```

Now, we provide a projector onto the computational and ancillary subspaces

```@example test1
Pc = σI # The projector onto the computational subspace (which is the whole Hilbert space in this case)
Pa = 0.0*Pc # The projector onto the ancillary subspaces (which is a null operator in this case)
prob = QOCProblem(U_target, T, Pc, Pa) # The quantum optimal control problem. 
```

## Defining a reduction map

A reduction map defines how the output from the Schrodinger equation (SE) and the GOAT EOMs are manipulated to return the objective function and any associated derivatives. Here we will use an already implemented reduction map that computes a projective SU measure called "infidelity."

```@example test1
SE_reduce_map = SE_infidelity_reduce_map 
GOAT_reduce_map = GOAT_infidelity_reduce_map
```

## Specifying solver and optimization options

```@example test1
# Define options for DifferentialEquations.jl (see DifferentialEquation docs for info)
diffeq_options = (abstol = 1e-9, reltol= 1e-9, alg=Vern9())

# Define the optimizer options from Optim.jl (See Optim.jl docs for info)
optim_alg = Optim.LBFGS(linesearch=LineSearches.BackTracking()) # A Back-Tracking linesearch
optim_options = Optim.Options(g_tol=1e-12,
                            iterations=10,
                            store_trace=true,
                            show_trace=true, extended_trace=false, allow_f_increases=false)
```

# Define the initial guess and perform an optimization

Here we will define our initial guess and specify which parameters are to be optimized, and how parallelization is to proceed:

```@example test1
p0 = [0.5,T/2,T/8] # The initial parameter guesses
opt_param_inds = [1] # The parameters of the vector p0 to optimize (just the amplitude parameter -- p0[1] -- in this case)
num_params_per_GOAT = 1 
```

This `num_params_per_GOAT` variable specifies how many derivatives are propogated in each GOAT EOMs and informs parallelization. 

For example, if `num_params_per_GOAT=5`and there are 5 total parameters, then no parallelization is performed. In contrast, if `num_params_per_GOAT=2` and the  re are 5 total parameters, then 3 processes are run in parallel: the first processor computes the EOMs for 2 parameters, the second process computes the EOMs for 2 parameters, and the third computes the EOMs for 1 parameter. 

Finally we run our optimization

```@example test1
res = find_optimal_controls(p0, opt_param_inds,sys, prob, 
                            SE_reduce_map, GOAT_reduce_map, diffeq_options, 
                            optim_alg, optim_options; num_params_per_GOAT = 1) 
```