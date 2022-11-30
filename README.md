# Gradient Optimization of Analytic Controls in Julia

This project is a [Julia](https://julialang.org/) implementation of the Gradient Optimization of Analytic conTrols (GOAT) optimal control methodology proposed in [this paper](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.150401) by Machnes et al.

The implementation only currently supports optimizing unitary evolution. 

## Overview of the GOAT method

The task of unitary quantum optimal control is to determine a set of control parameters $\vec{\alpha}$ that realize a desired unitary evolution on the quantum system. 

Specifically, the dynamics under the Hamiltonian

$$
H(\vec{\alpha},t)  = H_0 + \sum_i c_i(\vec{\alpha},t) H_i
$$

generates a unitary evolution

$$
U(\vec{\alpha},t) = \mathcal{T}\exp \bigg[ \int_0^t d\tau H(\vec{\alpha},\tau) \bigg].
$$

The task of optimal control is to determine a set of control parameters $\vec{\alpha}^*$ and a control time $T^*$ such that the system's evolution $U(\vec{\alpha}^*, T^*)$ approximates a desired unitary evolution $U_{target}$. 

The GOAT algorithm assumes that the each control function is described as a combination of a set of parametrized basis functions. This is a rather general definition and many choices for this decomposition can be made. Referred to as an *ansatz*, the choice of decomposition is typically informed by experimental considerations. One example is the expansion of each control function within an orthogonal function basis (like a polynomial basis or Fourier basis):

$$
c_i(\vec{\alpha},t) = \sum_n^N \alpha_n f_n(t).
$$

While this optimization can be performed using gradient-free methods, optimization algorithms that utilize information about the gradient of the objective function w.r.t. the control parameters can yield much faster optimization. 

In order to utilize gradient-based optimization routines it is required that we compute the partial derivative of the unitary evolution with respect to a particular parameter $\alpha_n$: $\partial_{\alpha_n} U({\vec{\alpha},T})$. This can then be used (via the chain rule) to compute gradients of the objective function. 

We compute $\partial_{\alpha_n} U({\vec{\alpha},T})$ by deriving a new differential equation from the Schrodinger equation:

$$
i\hbar \partial_t U(\vec{\alpha},t) = H(\vec{\alpha},t)U(\vec{\alpha},t)
$$

by differentiating w.r.t $\alpha_n$ and utilizing the product rule

$$
\begin{align*}
i\hbar\partial_{\alpha_n} \partial_t U(\vec{\alpha},t) &= \bigg(\partial_{\alpha_n} H(\vec{\alpha},t) \bigg) U(\vec{\alpha},t) + \bigg( H(\vec{\alpha},t) \bigg) \partial_{\alpha_n}U(\vec{\alpha},t)\\
&= i\hbar\partial_t \partial_{\alpha_n} U(\vec{\alpha},t).
\end{align*}
$$

The last step utilizes the symmetry of the second derivative to obtain a differential equation for $\partial_{\alpha_n} U(\vec{\alpha},t)$. 

To compute $\partial_{\alpha_n} U(\vec{\alpha},T)$ at a final time $T$, we integrate the coupled differential equations:

$$

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

$$

This coupled differential equation is referred to as the GOAT Equations of Motion (EOM). It has a significant amount of internal structure which is specialized on in this package for efficient computation.

First, it depends only on a single parameter $\alpha_n$ thus, computing the gradient can be obtained through parallelization, solving a seperate GOAT EOM for each parameter to be optimized. 

Second, the action of the differential equation is highly structured: the upper triangle of the matrix is $0$ and the diagonal of the matrix is $H(\vec{\alpha},t)$. Thus, storing this matrix, even in a sparse form is inefficient. Not only that, but the Hamiltonians of most physically realizable quantum systems are sparse which suggests that instantiating the action of the EOMs is not efficient in general. 

Thus, this package specializes on these structures to implement parallelized, matrix-free implementations of the GOAT EOMs to optimize performance.

## Package structure and use

The GOAT.jl package aims to provide an all-in-one implementation of GOAT for users within the fields of quantum physics and quantum computation. This means that a user can specify a controllable quantum system, a quantum optimal control problem, a control ansatz, and perform optimization. 

The main functionality of this package is centered on two data structures, the `ControllableSystem` and the `QOCProblem`.  Once these structures are instantiated GOAT.jl provides a number of methods to iteratively solve the GOAT EOMs (via [DifferentialEquations.jl](https://diffeq.sciml.ai/stable/)) and determine controls through optimization (via [Optim.jl](https://julianlsolvers.github.io/Optim.jl/stable/)).

However, one does not need to use DifferentialEquations.jl or Optim.jl. At it's core, GOAT.jl uses a `ControllableSystem` struct to generate efficient functions for the [in-place](https://diffeq.sciml.ai/stable/basics/problem/#In-place-vs-Out-of-Place-Function-Definition-Forms) definition of a differential equation. For more details on utilizing custom optimization or numerical integration please see the examples. 

While not a direct dependency, the authors recommend using the [QuantumOptics.jl](https://qojulia.org/) to model the details of the quantum system (vectors/operators used in the computation). At the moment, GOAT.jl does not have methods that accept operator types from QuantumOptics.jl, but we do plan to add this functionality in the future. 

## Example 1: Optimizing a gaussian pulse for a single qubit

This is perhaps the simplest example of a quantum optimal control problem. The task is to determine a the amplitude of a Guassian-shaped pulse such that 

## Example 2: Optimizing a custom pulse-ansatz for a transmon using QuantumOptics.jl




