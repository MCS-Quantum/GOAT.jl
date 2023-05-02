# Finding optimal controls

Once the primary datastructures have been defined (`ControllableSystem`, `QOCProblem`, and `QOCParams`) we are now in a position to perform the quantum optimal control experiment!! This is done as follows:

```julia
sys = ControllableSystem(...)
prob = QOCProblem(...)
params = QOCParams(...)
p0 = [...] # Initial guess
result = find_optimal_controls(p0, sys, prob, params)
```

which will begin the optimization at the initial guess `p0` and continue until the stopping criteria is reached. 

This syntax is useful because it allows the user to modify systems, problems, parameters, or initial guesses independently of one another. For example, one could re-define the `ControllableSystem` using a different ansatz or reference frame and not have to change any other bit of code. This is particularly useful if one is interested in custom control ansatze or objective functions. 