# Gradient Optimization of Analytic Controls in Julia

![GOAT_logo](/docs/build/assets/logo.svg)

This project is a [Julia](https://julialang.org/) implementation of the Gradient Optimization of Analytic conTrols (GOAT) optimal control methodology proposed in [this paper](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.150401) by Machnes et al.

The implementation only currently supports optimizing unitary evolution. 

## Overview of the GOAT method

The task of unitary quantum optimal control is to determine a set of control parameters $`\vec{\alpha}`$ that realize a desired unitary evolution on the quantum system. 

Specifically, the dynamics under the Hamiltonian

```math
H(\vec{\alpha},t)  = H_0 + \sum_i c_i(\vec{\alpha},t) H_i
```

generates a unitary evolution

```math
U(\vec{\alpha},t) = \mathcal{T}\exp \bigg[ \int_0^t d\tau H(\vec{\alpha},\tau) \bigg].
```

The task of optimal control is to determine a set of control parameters $`\vec{\alpha}^*`$ and a control time $`T^*`$ such that the system's evolution $`U(\vec{\alpha}^*, T^*)`$ approximates a desired unitary evolution $`U_{target}`$. 

The GOAT algorithm assumes that the each control function is described as a combination of a set of parametrized basis functions. This is a rather general definition and many choices for this decomposition can be made. Referred to as an *ansatz*, the choice of decomposition is typically informed by experimental considerations. One example is the expansion of each control function within an orthogonal function basis (like a polynomial basis or Fourier basis):

```math
c_i(\vec{\alpha},t) = \sum_n^N \alpha_n f_n(t).
```

While this optimization can be performed using gradient-free methods, optimization algorithms that utilize information about the gradient of the objective function w.r.t. the control parameters can yield much faster optimization. 

In order to utilize gradient-based optimization routines it is required that we compute the partial derivative of the unitary evolution with respect to a particular parameter $`\alpha_n`$: $`\partial_{\alpha_n} U({\vec{\alpha},T})`$. This can then be used (via the chain rule) to compute gradients of the objective function. 

We compute $`\partial_{\alpha_n} U({\vec{\alpha},T})`$ by deriving a new differential equation from the Schrodinger equation:

```math
i\hbar \partial_t U(\vec{\alpha},t) = H(\vec{\alpha},t)U(\vec{\alpha},t)
```

by differentiating w.r.t $`\alpha_n`$ and utilizing the product rule

```math
\begin{align*}
i\hbar\partial_{\alpha_n} \partial_t U(\vec{\alpha},t) &= \bigg(\partial_{\alpha_n} H(\vec{\alpha},t) \bigg) U(\vec{\alpha},t) + \bigg( H(\vec{\alpha},t) \bigg) \partial_{\alpha_n}U(\vec{\alpha},t)\\
&= i\hbar\partial_t \partial_{\alpha_n} U(\vec{\alpha},t).
\end{align*}
```

The last step utilizes the symmetry of the second derivative to obtain a differential equation for $`\partial_{\alpha_n} U(\vec{\alpha},t)`$. 

To compute $`\partial_{\alpha_n} U(\vec{\alpha},T)`$ at a final time $`T`$, we integrate the coupled differential equations:

```math

\partial_t
\begin{pmatrix}
U(\vec{\alpha},t)\\
\partial_{\alpha_n} U(\vec{\alpha},t)
\end{pmatrix}

= 

\begin{pmatrix}
H(\vec{\alpha},t) & 0\\
\partial_{\alpha_n} H(\vec{\alpha},t) & H(\vec{\alpha},t)
\end{pmatrix}

\begin{pmatrix}
U(\vec{\alpha},t)\\
\partial_{\alpha_n} U(\vec{\alpha},t)
\end{pmatrix}.

```

This coupled differential equation is referred to as the GOAT Equations of Motion (EOM). The initial conditions are $`U(\vec{\alpha},0) = I`$ and $`\partial_{\alpha_n} U(\vec{\alpha},0) = 0~ ~ \forall ~ ~ \alpha_n`$. It has a significant amount of internal structure which is specialized on in this package for efficient computation.

First, it depends only on a single parameter $`\alpha_n`$ thus, computing the gradient can be obtained through parallelization, solving a seperate GOAT EOM for each parameter to be optimized. 

Second, the action of the differential equation is highly structured: the upper triangle of the matrix is null and the diagonal of the matrix is $`H(\vec{\alpha},t)`$. Thus, storing this matrix, even in a sparse form is inefficient. Not only that, but the Hamiltonians of most physically realizable quantum systems are sparse which suggests that instantiating the action of the EOMs is not efficient in general. 

Thus, this package specializes on these structures to implement parallelized, matrix-free implementations of the GOAT EOMs to optimize performance.

## Package structure and use

The GOAT.jl package aims to provide an all-in-one implementation of GOAT for users within the fields of quantum physics and quantum computation. This means that a user can specify a controllable quantum system, a quantum optimal control problem, a control ansatz, and perform optimization. 

The main functionality of this package is centered on two data structures, the `ControllableSystem` and the `QOCProblem`.  Once these structures are instantiated GOAT.jl provides a number of methods to iteratively solve the GOAT EOMs (via [DifferentialEquations.jl](https://diffeq.sciml.ai/stable/)) and determine controls through optimization (via [Optim.jl](https://julianlsolvers.github.io/Optim.jl/stable/)).

However, one does not need to use DifferentialEquations.jl or Optim.jl. At it's core, GOAT.jl uses a `ControllableSystem` struct to generate efficient functions for the [in-place](https://diffeq.sciml.ai/stable/basics/problem/#In-place-vs-Out-of-Place-Function-Definition-Forms) definition of a differential equation.

### Reference frames

Within this package we also provide the support to define dynamical reference frames. A choice of reference frame (sometimes called an "rotating frame" or an "interaction picture") is a mathematical and physical utility used to simplify quantum dynamics. 

A dynamical reference frame is defined by a Hermitian operator: $`A`$ (we currently only support time-independent reference frames but plan to add more functionality in the future). This operator may be derived from the system's Hamiltonian and is often chosen in such a way that the effective quantum dynamics within the reference frame is simplified. The most common example taught in contemporary quantum mechanics is known as the [interaction picture](https://en.wikipedia.org/wiki/Interaction_picture). 

However, the interaction picture is only a particular choice of refence frame! Many other references frames can be defined which are esspecially interesting or useful (See for example, the original [DRAG paper](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.103.110501) where a time-dependent reference frame is used to derive some of the most effective and common control pulses used today.)

Reference frames are particularly important in experimental quantum devices. One example of this is occurs in assessing the "idling" error. Idling error occurs when coherent quantum dynamics occur even when the system is not being controlled, i.e. there is a component of the Hamiltonian that is non-zero at every time. This leads to decoherence and errors. A reference frame can be chosen to mininimize these unwanted errors. Thus, all controls need to be chosen such that they generate the desired evolutions within the specified reference frame. 

For example, in experimental systems local clocks are used to "lock" a reference frame and synchronize classical electronics. If there is noise in the clock this appears as noise within the quantum dynamics and can affect quantum device performance (See [this paper](https://www.nature.com/articles/npjqi201633)). 

### Recommendations

While not a direct dependency, the authors recommend using the [QuantumOptics.jl](https://qojulia.org/) to model the details of the quantum system (vectors/operators used in the computation). At the moment, GOAT.jl does not have methods that accept operator types from QuantumOptics.jl, but we do plan to add this functionality in the future. 

## Example 1: Optimizing a gaussian pulse for a single qubit

This is perhaps the simplest example of a quantum optimal control problem. The physical task is to invert a qubit's population from he ground state $`\ket{0}`$ to the orthogonal state $`\ket{1}`$.

Given a qubit Hamiltonian 

```math
H(t) = \frac{\omega}{2}\sigma_x + \Omega(t) \cos(\omega t)\sigma_x
```

one can determine analytically that there are a family of pulses which will accomplish the control task if the following topological criteria is met:

```math
\int_0^t d\tau \Omega(\tau) = \pi
```

one common choice of $`\Omega(t)`$ is a Gaussian function (due to an array of reasons outside the scope of this example). 

For this example we will fix the mean and standard deviation of the Gaussian to $`\mu=T/2`$ and $`\sigma = T/6`$. Therefore, we must determine the amplitude of the Gaussian in order to satisfy the topological criteraia and create population inversion.

The code to accomplish this is below:

```julia
using GOAT
using LinearAlgebra, SparseArrays # For instantiating necessary matrices
using DifferentialEquations # Load differetial equation algorithms
using Optim, LineSearches # Load optimization and linesearch algorithms

σx = sparse(ComplexF64[0 1 ; 1  0])# The Pauli X matrix
σy = sparse(ComplexF64[0 -im ; im 0]) # The Pauli Y matrix
σz = sparse(ComplexF64[1 0 ; 0 -1]) # The Pauli Z matrix
σI = sparse(ComplexF64[1 0 ; 0  1]) # The identity matrix
ω = 5*2pi # Setting a qubit frequency
T = 10.0 # in arbitrary time units
H0 = (ω/2)*σz # Setting the drift component of the Hamiltonian
basis_ops = [σx] # The control basis operator
Ω(p,t,i) = gaussian_coefficient(p,t,i,1)*cos(ω*t) # The control ansatz is a gaussian envelope of a cosine wave at frequency ω
∂Ω(p,t,i,l) = ∂gaussian_coefficient(p,t,i,l,1)*cos(ω*t) # The derivative of the control ansatz
rotating_frame_generator = -Diagonal(Matrix(H0)) # Moving into the reference frame rotating at the qubit frequency

U_target = deepcopy(σx) # The targt unitary: X|0> = |1>

p0 = [0.5,T/2,T/8] # The initial parameter guesses
sys = ControllableSystem(H0, basis_ops, rotating_frame_generator, Ω, ∂Ω)

Pc = σI # The projector onto the computational subspace (which is the whole Hilbert space in this case)
Pa = 0.0*Pc # The projector onto the ancillary subspaces (which is a null operator in this case)
prob = QOCProblem(U_target, T, Pc, Pa) # The quantum optimal control problem. 

# Defining a reduction map that takes the output from the Schrodinger equation (SE) and returns an infidelity
# Here we will use an already implemented reduction map, but in principle any custom map can be defined
SE_reduce_map = SE_infidelity_reduce_map 

# Defining a reduction map that takes the output from the GOAT EOMs and returns an infidelity 
# and it's gradient w.r.t a parameter
# Here we will use an already implemented reduction map, but in principle any custom map can be defined
GOAT_reduce_map = GOAT_infidelity_reduce_map

# Define options for DifferentialEquations.jl (see DifferentialEquation docs for info)
diffeq_options = (abstol = 1e-9, reltol= 1e-9, alg=Vern9())

# Define the optimizer options from Optim.jl (See Optim.jl docs for info)
optim_alg = Optim.LBFGS(linesearch=LineSearches.BackTracking()) # A Back-Tracking linesearch
optim_options = Optim.Options(g_tol=1e-12,
                            iterations=10,
                            store_trace=true,
                            show_trace=true, extended_trace=false, allow_f_increases=false)

opt_param_inds = [1] # The parameters of the vector p0 to optimize (just the amplitude parameter -- p0[1] -- in this case)
num_params_per_GOAT = 1 # This parameter specifies how many derivatives are propogated in each GOAT EOMs
# and informs parallelization. For example, if num_params_per_GOAT=5 and there are 5 total parameters,
# then no parallelization is performed. In contrast, if num_params_per_GOAT=2 and there are 5 total parameters,
# then 3 processes are run in parallel: the first processor computes the EOMs for 2 parameters,
# the second process computes the EOMs for 2 parameters, and the third computes the EOMs for 1 parameter. 
res = find_optimal_controls(p0, opt_param_inds,sys, prob, 
                            SE_reduce_map, GOAT_reduce_map, diffeq_options, 
                            optim_alg, optim_options; num_params_per_GOAT = 1) 




```

## Example 2: Optimizing a custom pulse-ansatz for a transmon using QuantumOptics.jl

In this example we will utilize QuantumOptics.jl to define a more complex quantum system -- a Duffing oscillator (a good model of a [Transmon](https://qiskit.org/textbook/ch-quantum-hardware/transmon-physics.html)). We will also define a custom pulse ansatz using some of the pre-defined basis functions in GOAT.jl.

The quantum systems that people deal with in a laboratory are very often *not* qubits. Instead, one defines a qubit subspace within the quanutm system you have access to. For example, a duffing oscillator has the Hamiltonian:

```math
H = \omega \hat{n}+\frac{\delta}{2} \hat{n}(\hat{n}-1)
```

where the frequency of the oscillator is specified by $`\omega`$ and $`\delta`$ defines the anharmonicity (it adds a perturbation that makes different energy levels of the oscillator have different transition frequencies). 

This system has an infinite number of eigenstates that form a ladder-like structure. However, we can define a "qubit" within a 2-dimensional subspace. For instance, we can choose the subspace spanned by the two lowest energy eigenstates: $`\{|0 \rangle, |1\rangle \}`$. 

In this example, we will use GOAT.jl to find a optimal control that implements a $`\pi`$ rotation within this subspace (i.e, find a unitary $`V`$ s.t. $`V|0\rangle \propto |1\rangle`$ and $`V|1\rangle \propto |0\rangle`$). This is a much more challenging task than the one explored in Example 1. In particular, we must make sure that the unitary $`V`$ does not connect (or minimially connects) the $`|0\rangle`$ or $`|1\rangle`$ states to any other states of the oscillator. This is a process called leakage. 

Our control term has the form 

```math
H_c(t) = \Omega(t)(\hat{a} + \hat{a}^\dagger). 
```

Where $`\hat{a}`$ and $`\hat{a}^\dagger`$ are the excitation [anhilation and creation operators](https://en.wikipedia.org/wiki/Creation_and_annihilation_operators), respectively. 

```julia
using GOAT
using QuantumOptics
using LinearAlgebra, SparseArrays # For instantiating necessary matrices
using DifferentialEquations # Load differetial equation algorithms
using Optim, LineSearches # Load optimization and linesearch algorithms

# Use QuantumOptics to define a more complex quantum system
b = FockBasis(3) # A Hilbert space basis using a particle-number cutoff of 3 : span(|0⟩, |1⟩, |2⟩, |3⟩)

a = destroy(b) # The particle anhilation operator
at = create(b) # The particle creation operator
n = number(b) # The particle number operator
Id = identityoperator(b) # The identity operator

ω = 5*2pi # Setting a oscillator frequency
δ = -0.2*2pi # Setting the oscillator anharmonicity 
T = 20.0 # in arbitrary time units
H0 = (ω*n+ 0.5*δ*n*(n-Id)).data # Setting the drift component of the Hamiltonian 
# (we access the underlying matrix of the QuantumOptics operator using ".data"
Hc = (a+at).data
basis_ops = [Hc] # The operators which define the control basis

N = 1 # The number of basis functions to use to describe the control. 
# Each ansatz has a different number of parameters for each basis function
# For the Fourier basis the parameters are the amplitude, frequency, and phase of each sinusoid
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

rotating_frame_generator = -Diagonal(Matrix(H0)) # Moving into the reference frame rotating at the oscillator frequencies

Pc = (projector(fockstate(b,0))+projector(fockstate(b,1))).data # The projector onto the computational subspace 
# (which is the first two energy levels in this case: |0⟩, |1⟩)
# We define this opreator because in some sense, we don't care what evolution the other subspaces experience
# as long as the evolution within the subspace we are interested in is what we want it to be
Pa = 0.0*Id.data # The projector onto the ancillary subspaces (which is a null operator in this case)
# This projector can be used to penalize population on higher energy states
# but we won't consider that in this example

U_target = Matrix(Id.data) # We begin by initializing a matrix
U_target[1,1] = 0.0 
U_target[2,2] = 0.0
U_target[1,2] = 1.0
U_target[2,1] = 1.0 # We now have a target unitary that acts nontrivially only on the computational subspace.

p0 = [0.25,ω,0.0] # The initial parameter guesses
sys = ControllableSystem(H0, basis_ops, rotating_frame_generator, Ω, ∂Ω)

prob = QOCProblem(U_target, T, Pc, Pa) # The quantum optimal control problem. 

# Defining a reduction map that takes the output from the Schrodinger equation (SE) and returns an infidelity
# Here we will use an already implemented reduction map, but in principle any custom map can be defined
SE_reduce_map = SE_infidelity_reduce_map 

# Defining a reduction map that takes the output from the GOAT EOMs and returns an infidelity 
# and it's gradient w.r.t a parameter
# Here we will use an already implemented reduction map, but in principle any custom map can be defined
GOAT_reduce_map = GOAT_infidelity_reduce_map

# Define options for DifferentialEquations.jl (see DifferentialEquation docs for info)
diffeq_options = (abstol = 1e-9, reltol= 1e-9, alg=Vern9())

# Define the optimizer options from Optim.jl (See Optim.jl docs for info)
optim_alg = Optim.LBFGS(linesearch=LineSearches.BackTracking()) # A Back-Tracking linesearch
optim_options = Optim.Options(g_tol=1e-12,
                            iterations=20,
                            store_trace=true,
                            show_trace=true, extended_trace=false, allow_f_increases=false)

opt_param_inds = [1,2,3] # The parameters of the vector p0 to optimize -- all parameters in this case
num_params_per_GOAT = 3 # This parameter specifies how many derivatives are propogated in each GOAT EOMs
# and informs parallelization. For example, if num_params_per_GOAT=5 and there are 5 total parameters,
# then no parallelization is performed. In contrast, if num_params_per_GOAT=2 and there are 5 total parameters,
# then 3 processes are run in parallel: the first processor computes the EOMs for 2 parameters,
# the second process computes the EOMs for 2 parameters, and the third computes the EOMs for 1 parameter. 
res = find_optimal_controls(p0, opt_param_inds,sys, prob, 
                            SE_reduce_map, GOAT_reduce_map, diffeq_options, 
                            optim_alg, optim_options; num_params_per_GOAT = 1) 
# The converged infidelity should be around 7e-4, higher fidelities are possible. Explore the ansatz, control time, and other parameters to find a better optimum!

```







