# Gradient Optimization of Analytic Controls in Julia

<p align="center">
<img class="GOAT_logo" width="300" height="300" src="https://github.com/MCS-Quantum/GOAT.jl/blob/main/docs/src/assets/logo.svg" alt="GOAT.jl Logo">
</p>

This project is a [Julia](https://julialang.org/) implementation of the Gradient Optimization of Analytic conTrols (GOAT) optimal control methodology proposed in [this paper](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.150401) by Machnes et al.

The implementation only currently supports optimizing unitary evolution. 

## Installation

```julia
] add OrdinaryDiffEq # Optionally install DifferentialEquations.jl for more functionality, customization, and analysis
] add Optim, LineSearches
] add https://github.com/MCS-Quantum/GOAT.jl # Until I get the project registered
## Optional - Useful for modeling various quantum systems
] add QuantumOptics
```

## Documentation

[Please see the documentation and examples](https://mcs-quantum.github.io/GOAT.jl/stable/) for how to use and improve this software!

## Citation

Please cite this research using the following citation:

