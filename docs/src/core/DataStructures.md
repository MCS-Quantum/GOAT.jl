# Core data structures

## ControllableSystem

Performing a quantum optimal control simulation requires various inputs and options. We must specify the quantum system and how it is controlled, specify the optimal control problem objective, and we define the methods and parameters for both the optimization routines and the solution of the differential equations. 

In GOAT.jl we specify all of this required information into three core structures: `ControllableSystem`, `QOCProblem`, and `QOCParameters`. 

The first is `ControllableSystem`, which holds all the information about how the quantum system evolves with a given set of control parameters and even holds information regarding the reference frame that the evolution is being computed in. We do not anticipate most users to delve into the structure of `ControllableSystem` so we provide a number of methods to generate this structure in a less complex manner. 

As an example, consider a controllable system with a simple structure that we wish to use for gradient-based quantum optimal control:

```math
H(\vec{\alpha},t)  = H_0 + \sum_i c_i(\vec{\alpha},t) H_i
```

first we define the operators

```julia
drift = H0
control_basis_ops = [H1,H2,H3] # Just a vector of matrices
```

then we define the coefficient function and its derivative w.r.t. the control parameters (See Ansatze section). For simplicity we will assume that each coefficient is the sum of $N=3$ Gaussian functions, each with variable amplitude, mean, and standard deviation:

```julia
c_func(p,t,i,N) = gaussian_coefficient(p,t,i,3)
∂c_func(p,t,i,l,N) = gaussian_coefficient(p,t,i,l,3)
```

Then we will call the following method:

```julia
sys = ControllableSystem(H0,control_basis_ops,c_func,∂c_func)
```
This will automatically convert all of the arrays into a sparse format which will enable GOAT.jl to generate optimized functions for computing the GOAT EOMs. 


## QOCProblem

GOAT.jl provides the ability to solve quantum-process QOC tasks. These are tasks where the goal is to identify a set of controls that will cause the system to undergo a desired quantum process. This includes objectives where the goal is to generate a unitary evolution on the global Hilbert space, an arbitrary quantum process on a subspace, or perform initial-state specific tasks. 

The output from any solution to the Schrodinger equation or GOAT EOMs is a unitary operator on the full Hilbert space. The `QOCProblem` structure specifies what the objective function and the duration of the controls. 

For example, given a target unitary to implement `U_target` and a total control time `T` we can define the problem as:

```julia
Pc = σI # The projector onto the computational subspace (which is the whole Hilbert space in this case)
Pa = 0.0*Pc # The projector onto the ancillary subspaces (which is a null operator in this case)
prob = QOCProblem(U_target, T, Pc, Pa) # The quantum optimal control problem. 
```

## QOCParams

While `ControllableSystem` and `QOCProblem` define *what* the system is and *what* we are trying to accomplish via QOC, respectively, we also need to specify *how* the computation will proceed. That is where `QOCParams` comes into play.

Specifically, `QOCParams` has a relatively simple structure:

```julia
struct QOCParameters
    ODE_options
    SE_reduce_map
    GOAT_reduce_map
    optim_alg
    optim_options
    num_params_per_GOAT
end
```

`ODE_options` is a `NamedTuple` specifying the solver options for [OrdinaryDiffEq.jl](https://docs.sciml.ai/DiffEqDocs/stable/basics/common_solver_opts/), `SE_reduce_map` and `GOAT_reduce_map` are functions that take the output unitaries from solving the differential equations and map to the objective functions and gradients, `optim_alg` and `optim_options` are the options for [Optim.jl](https://julianlsolvers.github.io/Optim.jl/stable/#user/config/), and `num_params_per_GOAT` specifies how many gradients are propogated per CPU (effectively defining the parallelization of the QOC problem). 

Any user is strongly encouraged to view the examples for a detailed description of how to specify each of these parameters. 