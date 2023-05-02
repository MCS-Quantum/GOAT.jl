## The GOAT method

The task of unitary quantum optimal control is to determine a set of control parameters $\vec{\alpha}$ that realize a desired unitary evolution on the quantum system. 

Specifically, the dynamics under the Hamiltonian

```math
H(\vec{\alpha},t)  = H_0 + \sum_i c_i(\vec{\alpha},t) H_i
```

generates a unitary evolution

```math
U(\vec{\alpha},t) = \mathcal{T}\exp \bigg[ \int_0^t d\tau H(\vec{\alpha},\tau) \bigg].
```

The task of optimal control is to determine a set of control parameters $\vec{\alpha}^*$ and a control time $T^*$ such that the system's evolution $U(\vec{\alpha}^*, T^*)$ approximates a desired unitary evolution $U_{target}$. 

While this optimization can be performed using gradient-free methods, algorithms that utilize information about the gradient of the objective function w.r.t. the control parameters can yield much faster optimization. 

In order to utilize gradient-based optimization routines it is required that we compute the partial derivative of the unitary evolution with respect to a particular parameter $\alpha_n$: $\partial_{\alpha_n} U({\vec{\alpha},T})$. This can then be used (via the chain rule) to compute gradients of the objective function. 

We compute $\partial_{\alpha_n} U({\vec{\alpha},T})$ by deriving a new differential equation from the Schrodinger equation:

```math
i\hbar \partial_t U(\vec{\alpha},t) = H(\vec{\alpha},t)U(\vec{\alpha},t)
```

by differentiating w.r.t $\alpha_n$ and utilizing the product rule

```math
\begin{align*}
i\hbar\partial_{\alpha_n} \partial_t U(\vec{\alpha},t) &= \bigg(\partial_{\alpha_n} H(\vec{\alpha},t) \bigg) U(\vec{\alpha},t) + \bigg( H(\vec{\alpha},t) \bigg) \partial_{\alpha_n}U(\vec{\alpha},t)\\
&= i\hbar\partial_t \partial_{\alpha_n} U(\vec{\alpha},t).
\end{align*}
```

The last step utilizes the symmetry of the second derivative to obtain a differential equation for $\partial_{\alpha_n} U(\vec{\alpha},t)$. 

To compute $\partial_{\alpha_n} U(\vec{\alpha},T)$ at a final time $T$, we integrate the coupled differential equations:

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

This coupled differential equation is referred to as the GOAT equations of motion (EOMs). The initial conditions are $U(\vec{\alpha},0) = I$ and $\partial_{\alpha_n} U(\vec{\alpha},0) = 0~ ~ \forall ~ ~ \alpha_n$. It has a significant amount of internal structure which is specialized on in this package for efficient computation.

First, it depends only on a single parameter $\alpha_n$ thus, computing the gradient can be obtained through parallelization, solving a seperate GOAT EOM for each parameter to be optimized. 

Second, the action of the differential equation is highly structured: the upper triangle of the matrix is null and the diagonal of the matrix is $H(\vec{\alpha},t)$. Thus, storing this matrix, even in a sparse form is inefficient. Not only that, but the Hamiltonians of most physically realizable quantum systems are sparse which suggests that instantiating the action of the EOMs is not efficient in general. 

Thus, this package specializes on these structures to implement parallelized, matrix-free implementations of the GOAT EOMs to optimize performance. 

## Matrix-free implementation

Here we specify the matrix-free, in-place implementation of the GOAT equations of motion:

When solving the GOAT EOMs using [OrindaryDiffEq](https://github.com/SciML/OrdinaryDiffEq.jl) we need to generate a function `f(du,u,p,t)` that efficiently computes the derivative:

```math

\partial_t
\begin{pmatrix}
U(\vec{\alpha},t)\\
\partial_{\alpha_n} U(\vec{\alpha},t)
\end{pmatrix}

```

However, there are a number of ways to accelerate this, first we can peform the operation in-place, pre-allocating `du` and overwritting the matrix elements during the computation. This could be simply done via in-place matrix multiplication, however that would require allocating an array for the generator as a matrix at every time `t`:

```math

\begin{pmatrix}
H(\vec{\alpha},t) & 0\\
\partial_{\alpha_n} H(\vec{\alpha},t) & H(\vec{\alpha},t)
\end{pmatrix}

```

This is unnecessary because there are static components to the Hamiltonian and the generator of the GOAT EOMs is very redundant, as mentioned above. So, we specialize further and automatically generate [matrix-free methods](https://en.wikipedia.org/wiki/Matrix-free_methods) for `f!(du,u,p,t)` that do not allocate any intermediate arrays. Moreover, we minimize access to the pre-allocated arrays and eliminate redundant computation of the time-dependent coefficients by optimizing the in-place matrix multiplication methods. 