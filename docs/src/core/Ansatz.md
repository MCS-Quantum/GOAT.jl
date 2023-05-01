# Control Ansatze

## Defintion and syntax

The GOAT algorithm assumes that the each control function is described as a combination of a set of parametrized basis functions. Referred to as an *ansatz* (plural: *ansatze*), the choice of decomposition is typically informed by experimental considerations. GOAT.jl allows the user to specify arbitrary control ansatze and build complex ansatze that include window functions, saturation functions, filters, etc. 

Recall that we define the Hamiltonian as:

```math
H(\vec{\alpha},t)  = H_0 + \sum_i c_i(\vec{\alpha},t) H_i
```

The coefficient functions $c_i(\vec{\alpha},t)$ in GOAT.jl are a function of *all* the control parameters $\vec{\alpha}$, but only weight the control basis operator $H_i$. This is a mathematically general definition and any Hamiltonian can be specified in this way. However, it is common to see control functions that only depend on a small set of parameters.

From an interface point of view, the coefficient function must be defined as follows:

```julia
function c(p,t,i,N)
    ...
    return c
end
```

where `p` is the vector of parameters, `t` is the time, `i` defines which control operator $H_i$ it is associated with, and `N` indicates the number of basis functions used. One must also define the derivative in order to perform gradient-based optimization: 

```julia
function ∂c(p,t,i,l,N)
    ...
    return c
end
```

where `l` is the index that specifies which parameter `p[l]` that the differentiation is being performed about: `∂c(p,t,i,l,N) =` $\partial c/ \partial p_l |_t$. 

## Examples
Check out the examples to see how one can define custom ansatze and derivatives using the chain rule. 

## Included basis functions

Included in this package are a number of common function decompositions for a Fourier basis, Gaussian basis, and polynomial basis:

```@autodocs
Modules = [GOAT]
Pages = ["Ansatze.jl"]
Order   = [:function]
```

## Utility functions

We also include a number of utility functions that can be used when defining custom ansatze:

```@autodocs
Modules = [GOAT]
Pages = ["Utilities.jl"]
Order = [:function]
```
