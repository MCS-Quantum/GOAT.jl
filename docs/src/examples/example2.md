# Optimizing a custom pulse-ansatz for a transmon using QuantumOptics.jl

In this example we will utilize QuantumOptics.jl to define a more complex quantum system -- a Duffing oscillator (a good model of a [Transmon](https://qiskit.org/textbook/ch-quantum-hardware/transmon-physics.html)). We will also define a custom pulse ansatz using some of the pre-defined basis functions in GOAT.jl.

The quantum systems that people deal with in a laboratory are very often *not* qubits. Instead, one defines a qubit subspace within the quanutm system you have access to. For example, a duffing oscillator has the Hamiltonian:

```math
H = \omega \hat{n}+\frac{\delta}{2} \hat{n}(\hat{n}-1)
```

where the frequency of the oscillator is specified by $\omega$ and $\delta$ defines the anharmonicity (it adds a perturbation that makes different energy levels of the oscillator have different transition frequencies). 

This system has an infinite number of eigenstates that form a ladder-like structure. However, we can define a "qubit" within a 2-dimensional subspace. For instance, we can choose the subspace spanned by the two lowest energy eigenstates: $\{|0 \rangle, |1\rangle \}$. 

In this example, we will use GOAT.jl to find a optimal control that implements a $\pi$ rotation within this subspace (i.e, find a unitary $V$ s.t. $V|0\rangle \propto |1\rangle$ and $V|1\rangle \propto |0\rangle$). This is a much more challenging task than the one explored in Example 1. In particular, we must make sure that the unitary $V$ does not connect (or minimially connects) the $|0\rangle$ or $|1\rangle$ states to any other states of the oscillator. This is a process called leakage. 

Our control term has the form 

```math
H_c(t) = \Omega(t)(\hat{a} + \hat{a}^\dagger). 
```

Where $\hat{a}$ and $\hat{a}^\dagger$ are the excitation [anhilation and creation operators](https://en.wikipedia.org/wiki/Creation_and_annihilation_operators), respectively. 

## Imports 

```@example ex2
using GOAT
using QuantumOptics
using LinearAlgebra, SparseArrays # For instantiating necessary matrices
using OrdinaryDiffEq # Load differetial equation algorithms
using Optim, LineSearches # Load optimization and linesearch algorithms
```

## Defining the `ControllableSystem` using QuantumOptics.jl

We first define a Hilbert space using QuantumOptics and then the operators that live on that space:

```@example ex2
b = FockBasis(3) # A Hilbert space basis using a particle-number cutoff of 3 : span(|0⟩, |1⟩, |2⟩, |3⟩)
a = destroy(b) # The particle anhilation operator
at = create(b) # The particle creation operator
n = number(b) # The particle number operator
Id = identityoperator(b) # The identity operator
```

Next, we define the Hamiltonian parameters

```@example ex2
ω = 5*2pi # Setting a oscillator frequency
δ = -0.2*2pi # Setting the oscillator anharmonicity 
T = 20.0 # in arbitrary time units
```

And finally the Hamiltonian and the control basis operators. (we access the underlying matrix of the QuantumOptics operator using ".data" attribute)

```@example ex2
H0 = (ω*n+ 0.5*δ*n*(n-Id)).data # Setting the drift component of the Hamiltonian 
Hc = (a+at).data
basis_ops = [Hc] # The operators which define the control basis
```

### Defining a custom ansatz

We will now define a custom pulse ansatz consisting of a sum of sinusoid functions that are modulated with an envelope function that is a flat-topped-cosine. Each ansatz has a different number of parameters for each basis function. For the Fourier basis the parameters are the amplitude, frequency, and phase of each sinusoid. 

```@example ex2
N = 1 # The number of basis functions to use to describe the control. 
function Ω(p,t,i)
    envelope = flat_top_cosine(t,1,T,0.3*T) # A flat-top cosine envelope for our pulse
    c = fourier_coefficient(p,t,i,N) # The control ansatz is a sum of sinusoidal basis functions
    return c*envelope
end
    
function ∂Ω(p,t,i,l)
    envelope = flat_top_cosine(t,1,T,0.3*T) # A flat-top cosine envelope for our pulse
    ∂c = ∂fourier_coefficient(p,t,i,l,N) # The derivative of the control ansatz
    return ∂c*envelope
end
```

We now define the reference frame in which we will implement a desired unitary operator:

```@example ex2
rotating_frame_generator = -Diagonal(Matrix(H0)) # Moving into the reference frame rotating at the oscillator frequencies
```

And finally we define the `ControllableSystem`:

```@example ex2
sys = ControllableSystem(H0, basis_ops, rotating_frame_generator, Ω, ∂Ω)
```

## Defining the `QOCProblem`

In this example, we wish to generate a unitary operator within a subspace of the system's Hilbert space. This requires that the global unitary operator (which is what we compute by solving the Schrodinger Equation and GOAT EOMs) be block-diagonal with respect to a bipartion of the Hilbert space. We reinforce this constrating by specifying the projectors onto the computational and ancillary subspaces. 


```@example ex2
Pc = (projector(fockstate(b,0))+projector(fockstate(b,1))).data # The projector onto the computational subspace 
# (which is the first two energy levels in this case: |0⟩, |1⟩)
Pa = 0.0*Id.data # The projector onto the ancillary subspaces (which is a null operator in this case)
```

the projector `Pa` can be used to penalize population on higher energy states but we won't consider that in this example. There are a variety of ways to define the projector `Pa`. In this example a null operator was chosen because we don't care what evolution the other subspaces experience.

Next we define our target unitary operator (effectively a swap operation between states $\ket{0}$ and $\ket{1}$). The matrix elements of the unitary outside the computational subspace can be chosen arbitrarily because when computing the objective function they will be projected away by `Pc`. Here we just modify the identity matrix.

```@example ex2
U_target = Matrix(Id.data) # We begin by initializing a matrix
U_target[1,1] = 0.0 
U_target[2,2] = 0.0
U_target[1,2] = 1.0
U_target[2,1] = 1.0
```

Finally, we define the `QOCProblem`:

```@example ex2
prob = QOCProblem(U_target, T, Pc, Pa) # The quantum optimal control problem. 
```

## Defining a reduction map

A reduction map defines how the output from the Schrodinger equation (SE) and the GOAT EOMs are manipulated to return the objective function and any associated derivatives. Here we will use an already implemented reduction map that computes a projective SU measure called "infidelity."

```@example ex2
SE_reduce_map = SE_infidelity_reduce_map 
GOAT_reduce_map = GOAT_infidelity_reduce_map
```


## Specifying solver and optimization options

```@example ex2
# Define options for DifferentialEquations.jl (see DifferentialEquation docs for info)
ODE_options = (abstol = 1e-9, reltol= 1e-9, alg=Vern9())

# Define the optimizer options from Optim.jl (See Optim.jl docs for info)
optim_alg = Optim.LBFGS(linesearch=LineSearches.BackTracking()) # A Back-Tracking linesearch
optim_options = Optim.Options(g_tol=1e-12,
                            iterations=10,
                            store_trace=true,
                            show_trace=true, extended_trace=false, allow_f_increases=false)
```

# Define the initial guess and perform an optimization

Here we will define our initial guess and specify which parameters are to be optimized, and how parallelization is to proceed:

```@example ex2
p0 = [0.25,ω,0.0] # The initial parameter guesses
opt_param_inds = [1,2,3] # The parameters of the vector p0 to optimize -- all parameters in this case
derivs_per_core = 3 # This parameter specifies how many derivatives are propogated in each GOAT EOMs
```

This `derivs_per_core` variable specifies how many derivatives are propogated in each GOAT EOMs and informs parallelization. 

For example, if `derivs_per_core=5`and there are 5 total parameters, then no parallelization is performed. In contrast, if `derivs_per_core=2` and there are 5 total parameters, then 3 processes are run in parallel: the first processor computes the EOMs for 2 parameters, the second process computes the EOMs for 2 parameters, and the third computes the EOMs for 1 parameter. 

Next we put all of these parameters into a `QOCParameters` struct:

```@example ex2
params = QOCParameters(ODE_options,SE_reduce_map,GOAT_reduce_map,optim_alg,optim_options; derivs_per_core=derivs_per_core)
```

Finally we run our optimization

```@example ex2
res = find_optimal_controls(p0, opt_param_inds, sys, prob, params)
res
```

The converged infidelity should be around 7e-4, higher fidelities are possible. Explore the ansatz, control time, and other parameters to find a better optimum!
