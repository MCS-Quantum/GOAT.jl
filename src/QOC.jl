module QOC

using DifferentialEquations, NLopt, RecursiveArrayTools, LinearAlgebra, SparseArrays, FFTW, NumericalIntegration, QuantumOptics, DSP, Distributed, Random

export GOAT_objective!, objective!, g_sm, ∂g_sm, get_U_track, H!, ∂H!, linear_index_map, window, S, general_logistic, dSdx, LO, gaussian_kernel, fourier_ansatz, derivative_fourier_ansatz, derivative_gaussian_ansatz, gaussian_ansatz, sinusoid_kernel, solve_eoms, get_freqs_fft, h_sm, ∂h_sm, GOAT_objective_noleakage!, objective_noleakage!, opt_result_to_file, nlopt_optimization, get_periodogram, get_pair_objective_gradient, GOAT_objective_remote!, solve_pair_eoms, morlet_kernel, morlet_ansatz, derivative_morlet_ansatz, test_ansatz_derivatives, get_initial_guess, get_objective_func

function opt_result_to_file(filename, opt_object, minf, minx, ret)
    numevals = opt_object.numevals
    open(filename, "w") do f
        write(f, "Evaluations: $numevals\n")
        write(f, "Objective Value: $minf\n")
        write(f, "Termination Message: $ret\n")
        write(f, "$minx")
    end
end

function g_abs(U_target, U)
    @assert(size(U_target)==size(U),"Unitaries are different sizes")
    return 1-(1/size(U_target,1))*real(abs(tr(adjoint(U_target)*U)))
end

function ∂g_abs(U_target, ∂U)
    return (-1/size(U_target,1))*abs(tr(adjoint(U_target)*∂U))
end

function g_real(U_target,U)
    return 1-(1/size(U_target,1))*real(tr(adjoint(U_target)*U))
end

function ∂g_real(U_target,∂U)
    return (1/size(U_target,1))*real(tr(adjoint(U_target)*∂U))
end

function g_sm(U_target, U; dim=0)
    if dim==0
        dim = size(U_target,1)
    end
    @assert(size(U_target)==size(U),"Unitaries are different sizes")
    return 1-(1/dim^2)*real(abs2(tr(adjoint(U_target)*U)))
end

function ∂g_sm(U_target, U, ∂U; dim=0)
    if dim==0
        dim = size(U_target,1)
    end
    return (-2/(dim^2))*real(tr(adjoint(U_target)*∂U)*tr(adjoint(U)*U_target))
end

function L1(U, I1, I2)
    d1 = real(tr(I1))
    return (1/d1)*real(tr(I2*U*I1*adjoint(U)))
end

function h_sm(Us,Pc,Pd,ts)
    Tc = ts[end]
    tp = (x) -> L1(x,Pc,Pd)
    ys = tp.(Us)
    return (1/Tc)*integrate(ts,ys)
end

function ∂h_sm(Us,∂Us,I1,I2,ts)
    d1 = real(tr(I1))
    Tc = ts[end]
    tp = (x,∂x) -> real(tr( I1*adjoint(∂x)*I2*x*I1 + I1*adjoint(x)*I2*∂x*I1))
    ys = tp.(Us,∂Us)
    return (1/Tc)*(1/d1)*integrate(ts,ys)
end

function sinusoid_kernel(t, a, w, phi)
    return a*sin(w*t+phi)
end

function gaussian_kernel(t, a, mu,sigma)
    return a*exp(-0.5*( (t-mu) / sigma)^2)
end


"""
Morlet wavelet kernel (although not a true wavelet because it is not normalized):

morlet_kernel(t,a, mu, sigma, w, phi)
"""
function morlet_kernel(t,a, mu, sigma, w, phi)
    return a*exp(-0.5*( (t-mu) / sigma)^2)*sin(w*t+phi)
end

"""
A version of a generalized logistic function found on wikipedia: 
https://en.wikipedia.org/wiki/Generalised_logistic_function.
"""
function general_logistic(t, lower, upper, slope, start=1e3, nu=1.0, C=1)
    A = lower
    B = slope
    K = upper
    Q = start # The larger this is the smaller GL(0)
    return A + (K-A)/((C+Q*exp(-B*t))^(1/nu))
end


"""
A windowing function based on a product of two generalized logistic functions. 
"""
function window(x,lower,upper,gradient=20)
    rising = general_logistic(x-lower,0,1,gradient)
    lowering = general_logistic(-x+upper,0,1,gradient)
    return lowering*rising
end


"""
A saturation function that limits amplitudes to a particular range specified by [lower,upper].
"""
function S(x,lower,upper;gradient=1)
    mid = (upper-lower)*0.5
    Q = -upper/lower
    return general_logistic(x/mid,lower,upper,gradient,Q)
end


"""
Calculates the partial derivative of the saturation function w.r.t the independent variable x.
"""
function dSdx(x,lower,upper;gradient=5)
    mid = (upper-lower)*0.5
    Q = -upper/lower
    b = upper
    a = lower
    g = gradient
    xbar = x/mid
    return -( (b-a)/ ( (1+Q*exp(-g*xbar))^2 ))*(exp(-g*xbar))*(-g/mid)
end

"""
Adds a mixing effect for a local oscillator.
"""
function LO(t,w)
    return cos(w*t)
end


"""
Takes in a function for the signal, t0, tf, and the sampling period tp,
then returns the FT and frequencies in units of 1/[t] (Hz, MHz, GHz, for s, ms, ns). 
"""
function get_freqs_fft(et,t0,tf,tp)
    ts = [t0:tp:tf;]
    ets = et.(ts)
    F = fft(ets) |> fftshift
    freqs = fftfreq(length(ts),2*pi/(tp)) |> fftshift 
    return freqs,F
end
  

"""
Takes in a function for the signal, t0, tf, and the sampling period tp,
then returns the periodogram object with frequency in units of 1/[t] (Hz, MHz, GHz, for s, ms, ns). 
"""
function get_periodogram(et,t0,tf,tp)
    ts = [t0:tp:tf;]
    ets = et.(ts)
    p = periodogram(ets,fs=1/(tp))
    return p
end

function get_initial_guess(lbs,ubs, K, Ns, Qs; rand_func = rand)
    map = linear_index_map(Ns,Qs)
    M = sum(Qs.*Ns)
    x = zeros(Float64, M)
    for k in 1:K
        lbs_k = lbs[k]
        ubs_k = ubs[k]
        for n in 1:Ns[k]
            for q in 1:Qs[k]
                x[map[k,n,q]] = rand_func(Float64)*(ubs_k[q]-lbs_k[q])+lbs_k[q]
            end
        end
    end
    return x
end


"""
Returns a Hamiltonian with a control parameteriztion based on the provided kernel. 

The p.params vector must be in the same order as the control channel vector.
The p.params[1:M] should be in the order of the inputs to the kernel function.

It operates in-place on the temporary matrix in the field p.temp_matrix of the named tuple p.
"""
function H!(p,t; temp_matrix=nothing)
    
    drift = p.drift
    ansatz = p.ansatz
    if temp_matrix==nothing
        temp_matrix = p.temp_matrix
    end
    mul!(temp_matrix, drift,I, 1.0, 0.0) # Fill the temp matrix with the drift Hamiltonian
    D = size(temp_matrix,1)
    
    control_channels = p.channels # Control channels in VectorOfArray type ala RecursiveArrayTools.jl
    K = length(control_channels) # Number of control channels
    
    for k in 1:K
        c = ansatz(t,p,k) # Coefficient for channel k
        for i in 1:D
            for j in 1:D
                temp_matrix[i,j] += c*control_channels[i,j,k]
            end
        end
    end
    
    if :transform in keys(p)
        V = p.transform(t) # Must be a DiagonalMatrix
        Vt = adjoint(V) # Must be a DiagonalMatrix
        i∂VVt = p.transform_generator # Assume independent of t
        lmul!(V,temp_matrix) # VH
        rmul!(temp_matrix,Vt) # VHVt
        for i in 1:D
            for j in 1:D
                temp_matrix[i,j] += i∂VVt[i,j] #VHVt + i∂VVt
            end
        end
    end
    lmul!(-im,temp_matrix)
    
end


"""
Returns the partial derivative of the Hamiltonian with a control parameteriztion based on the provided kernel. 

The p.params vector must be in the same order as the control channel vector.
The p.params[1:M] should be in the order of the inputs to the kernel function.

It operates in-place on the temporary matrix in the field p.temp_matrix of the named tuple p.
"""
function ∂H!(p,t,m)

    map = p.map
    derivative_ansatz = p.derivative_ansatz
    
    temp_matrix = p.temp_matrix
    lmul!(0.0,temp_matrix) # Nullify the temp matrix
    D = size(temp_matrix,1)
    
    control_channels = p.channels # Control channels in VectorOfArray type ala RecursiveArrayTools.jl
    K = p.K # Number of control channels
    Ns = p.Ns # Number of basis functions for each channel
    Qs = p.Qs # Number of parameters per basis function for each channel
    
    k_set = 0
    n_set = 0
    q_set = 0
    
    for k in 1:K
        for n in 1:Ns[k]
            for q in 1:Qs[k]
                if map[k,n,q] == m
                    k_set += k
                    n_set += n
                    q_set += q
                end
            end
        end
    end
    
    c = derivative_ansatz(t,p,k_set,n_set,q_set) # Coefficient for paramter [k,n,q] = m
    for i in 1:D
        for j in 1:D
            temp_matrix[i,j] += c*control_channels[i,j,k_set]
        end
    end
    
    if :transform in keys(p)
        V = p.transform(t) # Must be a DiagonalMatrix
        Vt = adjoint(V) # Must be a DiagonalMatrix
        i∂VVt = p.transform_generator # Assume independent of t
        lmul!(V,temp_matrix) # VH
        rmul!(temp_matrix,Vt) # VHVt
    end
    lmul!(-im,temp_matrix)
end



function fourier_ansatz(t,p,k)
    a = p.params
    imap = p.map
    Ns = p.Ns
    c = 0.0
    # For the channel k the coefficient is defined by
    for n in 1:Ns[k]
        c += a[imap[k,n,1]]*sin(a[imap[k,n,2]]*t+a[imap[k,n,3]])
    end
    return c
end

function derivative_fourier_ansatz(t,p,k,n,q)
    a = p.params
    imap = p.map
    if q==1
        c = sin(a[imap[k,n,2]]*t+a[imap[k,n,3]])
        return c
    elseif q==2
        c = a[imap[k,n,1]]*t*cos(a[imap[k,n,2]]*t+a[imap[k,n,3]])
        return c
    else
        c = a[imap[k,n,1]]*cos(a[imap[k,n,2]]*t+a[imap[k,n,3]])
        return c
    end
end


function gaussian_ansatz(t,p,k)
    a = p.params
    imap = p.map
    Ns = p.Ns
    c = 0.0
    # For the channel k the coefficient is defined by
    for n in 1:Ns[k]
        c += a[imap[k,n,1]]*exp(-0.5*( (t-a[imap[k,n,2]]) / a[imap[k,n,3]])^2)
    end
    return c
end

function derivative_gaussian_ansatz(t,p,k,n,q)
    a = p.params
    imap = p.map
    if q==1
        c = exp(-0.5*( (t-a[imap[k,n,2]]) / a[imap[k,n,3]])^2)
        return c
    elseif q==2
        c = a[imap[k,n,1]]*(t-a[imap[k,n,2]])*exp(-0.5*( (t-a[imap[k,n,2]]) / a[imap[k,n,3]])^2) /(a[imap[k,n,3]]^2)
        return c
    else
        c = a[imap[k,n,1]]*((t-a[imap[k,n,2]])^2)*exp(-0.5*( (t-a[imap[k,n,2]]) / a[imap[k,n,3]])^2)/(a[imap[k,n,3]]^3)
        return c
    end
end


function morlet_ansatz(t,p,k)
    a = p.params
    imap = p.map
    N = p.Ns[k]
    c = 0.0
    # For the channel k the coefficient is defined by
    for n in 1:N
        c += a[imap[k,n,1]]*exp(-0.5*( (t-a[imap[k,n,2]]) / a[imap[k,n,3]])^2)*sin(a[imap[k,n,4]]*t+a[imap[k,n,5]])
    end
    return c
end

function derivative_morlet_ansatz(t,p,k,n,q)
    a = p.params
    imap = p.map
    if q==1
        c = exp(-0.5*( (t-a[imap[k,n,2]]) / a[imap[k,n,3]])^2)*sin(a[imap[k,n,4]]*t+a[imap[k,n,5]])
        return c
    elseif q==2
        c = a[imap[k,n,1]]*sin(a[imap[k,n,4]]*t+a[imap[k,n,5]])*(t-a[imap[k,n,2]])*exp(-0.5*( (t-a[imap[k,n,2]]) / a[imap[k,n,3]])^2) /(a[imap[k,n,3]]^2)
        return c
    elseif q==3
        c = a[imap[k,n,1]]*sin(a[imap[k,n,4]]*t+a[imap[k,n,5]])*((t-a[imap[k,n,2]])^2)*exp(-0.5*( (t-a[imap[k,n,2]]) / a[imap[k,n,3]])^2)/(a[imap[k,n,3]]^3)
        return c
    elseif q==4
        c = a[imap[k,n,1]]*exp(-0.5*( (t-a[imap[k,n,2]]) / a[imap[k,n,3]])^2)*t*cos(a[imap[k,n,4]]*t+a[imap[k,n,5]])
        return c
    else
        c = a[imap[k,n,1]]*exp(-0.5*( (t-a[imap[k,n,2]]) / a[imap[k,n,3]])^2)*cos(a[imap[k,n,4]]*t+a[imap[k,n,5]])
        return c
    end
end

function test_ansatz_derivatives(ansatz, derivative_ansatz, num_params; dx = 1e-8)
    test_params = rand(Float64,num_params)
    map = linear_index_map(1,1,num_params)
    p = (params=test_params, K = 1, N = 1, Q=num_params, map=map)
    t = rand(Float64)
    for q in 1:num_params
        p_mod = deepcopy(p)
        p_mod.params[map[1,1,q]] = p_mod.params[map[1,1,q]]+dx
        approx_deriv = (ansatz(t,p_mod,1)-ansatz(t,p,1))/dx
        deriv = derivative_ansatz(t,p,1,1,q)
        rel_diff = (approx_deriv - deriv)/approx_deriv
        print("q=$q: $rel_diff\n")
    end
end




function linear_index_map(Ns,Qs)
    keys = []
    vals = []
    K = length(Ns)
    i = 1
    M = zeros(Int64,K,maximum(Ns),maximum(Qs))
    for k in 1:K
        for n in 1:Ns[k]
            for q in 1:Qs[k]
                M[k,n,q] = i
                i+=1
            end
        end
    end
    return M
end




function GOAT_objective!(x::Vector, grad::Vector, p, tol,U_target)
    a = p.params
    a .= x
    M = length(a)
    d = size(p.temp_matrix,1)
    if :comp_dimension in keys(p)
        dc = p.comp_dimension
    else
        dc = d
    end
    #println(x)
    #println(p.a)
    Us = solve_eoms(p,tol)
    for i in 1:M
        grad[i] = ∂g_sm(U_target,Us[1],Us[i+1];dim=dc)
    end
    #println(grad)
    #println(x)
    ob = g_sm(U_target,Us[1];dim=dc)
    return ob
end

function GOAT_objective_noleakage!(x::Vector, grad::Vector, p, tol, U_target)
    a = p.params
    a .= x
    M = length(a)
    d = size(p.temp_matrix,1)
    Pc = p.Pc
    Pd = p.Pd
    leakage_weight = p.leakage_weight
    if :comp_dimension in keys(p)
        dc = p.comp_dimension
    else
        dc = d
    end
    #println(x)
    #println(p.a)
    sol = solve_eoms(p,tol,return_sol=true)
    ts = sol.t
    Us = [sol.u[i][1:d,:] for i in 1:length(ts)]
    ∂Us_temp = similar(Us)
    for i in 1:M
        for j in 1:length(ts)
            ∂Us_temp[j] = sol.u[j][i*d+1:i*d+d,:]
        end
        grad[i] = ∂g_sm(U_target,Pc*Us[end]*Pc,Pc*∂Us_temp[end]*Pc; dim=dc) + leakage_weight*∂h_sm(Us,∂Us_temp,Pc,Pd,ts)
    end
    #println(grad)
    #println(x)
    ob = g_sm(U_target,Pc*Us[end]*Pc; dim=dc) + leakage_weight*h_sm(Us,Pc,Pd,ts)
    return ob
end

function objective!(x::Vector, grad::Vector, p, tol, U_target)
    a = p.params
    a .= x
    M = length(a)
    d = size(p.temp_matrix,1)
    if :comp_dimension in keys(p)
        dc = p.comp_dimension
    else
        dc = d
    end
    #println(x)
    #println(p.a)
    Us, ts = get_U_track(x,p,tol)
    #for i in 1:M
    #    grad[i] = ∂g_sm(U_target,Us[1],Us[i+1])
    #end
    #println(grad)
    #println(x)
    ob = g_sm(U_target,Us[end]; dim=dc)
    return ob
end

function objective_noleakage!(x::Vector, grad::Vector, p, tol, U_target)
    a = p.params
    a .= x
    M = length(a)
    Pc = p.Pc
    Pd = p.Pd
    d = size(p.temp_matrix,1)
    leakage_weight = p.leakage_weight
    if :comp_dimension in keys(p)
        dc = p.comp_dimension
    else
        dc = d
    end
    #println(x)
    #println(p.a)
    Us, ts = get_U_track(x,p,tol)
    #for i in 1:M
    #    grad[i] = ∂g_sm(U_target,Us[1],Us[i+1])
    #end
    #println(grad)
    #println(x)
    ob = g_sm(U_target,Pc*Us[end]*Pc; dim=dc) + leakage_weight*h_sm(Us,Pc,Pd,ts)
    return ob
end

# function get_final_U(x::Vector, p, tol)
#     a = p.params
#     a .= x
#     Us = solve_eoms(p,tol)
#     return Us[1]
# end


"""
The update function for the full matrix for GOAT eoms.
"""
function update_func!(A,u,p,t)
    temp_matrix = p.temp_matrix
    d = size(temp_matrix,1) # Dimension of small arrays
    M = p.M
    D = size(A,1) # Dimension of A
    H_func = p.H_func
    ∂H_func = p.pH_func
    # First add the diagonal Hamiltonians
    H_func(p,t) # Get the Hamiltonian
    for i in 1:d:D
        #setindex!(A,temp_matrix,i:i+d-1,i:i+d-1)
        A[i:i+d-1,i:i+d-1] .= temp_matrix
    end
    # Check if we are evolving all the partial Us. If not, scale and return.
    if size(A,1)==size(p.temp_matrix,1)
        return
    end
    # Add the partial Hs to the first row
    for i in 1:M
        ∂H_func(p,t,i)
        q = i*d+1
        A[q:q+d-1,1:d] .= temp_matrix
    end
end

function solve_eoms(p,tol;return_sol=false)
    M = p.M
    tspan = p.tspan
    d = size(p.temp_matrix,1)
    u0 = zeros(ComplexF64,M*d+d,d)
    u0[1:d,1:d] .= Matrix(I,d,d)
    B = spzeros(ComplexF64,d*(M+1),d*(M+1)) # The matrix for the eoms.
    D = size(B,1)
    A = DiffEqArrayOperator(B,update_func=update_func!)
    prob = ODEProblem(A,u0,tspan,p)
    #sol = solve(prob,MagnusGL4(),dt=0.1,abstol=tol,reltol=tol)
    sol = solve(prob,abstol=tol,reltol=tol)
    if return_sol
        return sol
    end
    Us = [sol.u[end][i*d+1:i*d+d,:] for i in 0:M]
    return Us
end

function get_U_track(x::Vector, p, tol)
    a = p.params
    a .= x
    M = p.M
    tspan = p.tspan
    d = size(p.temp_matrix,1)
    u0 = zeros(ComplexF64,d,d)
    u0[1:d,1:d] .= Matrix(I,d,d)
    B = spzeros(ComplexF64,d,d) # The matrix for the eoms.
    A = DiffEqArrayOperator(B,update_func=update_func!)
    prob = ODEProblem(A,u0,tspan,p)
    sol = solve(prob,abstol=tol,reltol=tol)
    return sol.u, sol.t
end


function test_gradients(p)
    M = p.M
    Pc = p.Pc
    Pd = p.Pd
    U_target = p.U_target
    a0 = deepcopy(p.params)
    grads = similar(a0)
    GOAT_objective_remote!(a0,grads,p,1e-9)
    for i in 1:M
        x = deepcopy(a0)
        x[i] = x[i]+1e-8
        p1 = deepcopy(p)
        p1.params .= x
        analytic_gradient = grads[i]
        sol1 = solve_pair_eoms(p1,1e-9,M)
        sol = solve_pair_eoms(p,1e-9,M)
        g1 = g_sm(U_target,Pc*sol1.u[end][1:3,:]*Pc;dim=2)
        g2 = g_sm(U_target,Pc*sol.u[end][1:3,:]*Pc;dim=2)
        numeric_gradient = (g1-g2)/1e-8
        rel_diff = (analytic_gradient-numeric_gradient)/analytic_gradient
        println("Param $i : $rel_diff")
        #println("Param $i : $analytic_gradient, $numeric_gradient")
    end
end
"""
Perform a optimization with NLOPT.
"""
function nlopt_optimization(alg, opt_params, QOC_prob, x0)
    M = QOC_prob.M
    opt = Opt(alg,M)
    
    for key in keys(opt_params)
        setproperty!(opt, key, getfield(opt_params,key))
    end
    
    if alg in [:LD_MMA,:LD_CCSAQ]
        opt.params["verbosity"]=0
    end
    
    ## Run the optimization
    (minf,minx,ret) = NLopt.optimize(opt, x0)
    println("Optimization complete. Min objective: $minf")
    nes = opt.numevals
    #println("$ret_g, $nes")
    
    return minf, minx, ret, opt
    
end


function update_func_pair!(A,u,p,t)
    i = p.index[1]
    temp_matrix = p.temp_matrix # Temporary matrix
    d = size(temp_matrix,1) # Dimension of small arrays
    H_func = p.H_func # Hamiltonian function
    ∂H_func = p.pH_func # Partial Hamiltonian function
    # First add the diagonal Hamiltonians
    H_func(p,t) # Get the Hamiltonian
    A[1:d,1:d] .= temp_matrix
    A[d+1:end,d+1:end] .= temp_matrix
    # Add the partial H
    ∂H_func(p,t,i)
    A[d+1:end,1:d] .= temp_matrix
end

function update_func_single!(A,u,p,t)
    temp_matrix = p.temp_matrix # Temporary matrix
    d = size(temp_matrix,1) # Dimension of small arrays
    H_func = p.H_func # Hamiltonian function
    # First add the diagonal Hamiltonians
    H_func(p,t) # Get the Hamiltonian
    A[1:d,1:d] .= temp_matrix
end

function solve_pair_eoms(p,tol,alg,m)
    p.index[1] = m
    tspan = p.tspan
    d = size(p.temp_matrix,1)
    u0 = zeros(ComplexF64,2*d,d)
    u0[1:d,1:d] .= Matrix(I,d,d)
    B = spzeros(ComplexF64,d*2,d*2) # The matrix for the eoms.
    D = size(B,1)
    A = DiffEqArrayOperator(B,update_func=update_func_pair!)
    prob = ODEProblem(A,u0,tspan,p)
    #sol = solve(prob,MagnusGL4(),dt=0.1,abstol=tol,reltol=tol)
    sol = solve(prob,alg,abstol=tol,reltol=tol)
    return sol
end

function get_objective_func(x, p,tol,alg)
    p.params .= x
    tspan = p.tspan
    d = size(p.temp_matrix,1)
    u0 = zeros(ComplexF64,d,d)
    u0 .= Matrix(I,d,d)
    B = spzeros(ComplexF64,d,d) # The matrix for the eoms.
    D = size(B,1)
    A = DiffEqArrayOperator(B,update_func=update_func_single!)
    prob = ODEProblem(A,u0,tspan,p)
    #sol = solve(prob,MagnusGL4(),dt=0.1,abstol=tol,reltol=tol)
    sol = solve(prob,alg,abstol=tol,reltol=tol)
    Pc = p.Pc
    Pd = p.Pd
    leakage_weight = p.leakage_weight
    U_target = p.U_target
    if :comp_dimension in keys(p)
        dc = p.comp_dimension
    else
        dc = d
    end
    
    Us = sol.u
    g = g_sm(U_target,Pc*Us[end]*Pc; dim=dc) + leakage_weight*h_sm(Us, Pc, Pd, sol.t)
    sol=nothing
    finalize(sol)
    return g
end

function reduce_sol_pair_objective_gradient(p, sol, m)
    M = p.M
    Pc = p.Pc
    Pd = p.Pd
    d = size(p.temp_matrix,1)
    leakage_weight = p.leakage_weight
    U_target = p.U_target
    if :comp_dimension in keys(p)
        dc = p.comp_dimension
    else
        dc = d
    end
    
    Us = [sol.u[i][1:d,:] for i in 1:length(sol.t)]
    ∂Us = [sol.u[i][d+1:end,:] for i in 1:length(sol.t)]
    
    ∂g = ∂g_sm(U_target,Pc*Us[end]*Pc,Pc*∂Us[end]*Pc; dim=dc)
    ∂g += leakage_weight*∂h_sm(Us,∂Us,Pc,Pd,sol.t)
    if m == M
        g = g_sm(U_target,Pc*Us[end]*Pc; dim=dc) + leakage_weight*h_sm(Us, Pc, Pd, sol.t)
    else
        g = 0
    end
    return g, ∂g
end

function get_pair_objective_gradient(p,tol,alg,m)
    sol = solve_pair_eoms(p,tol,alg,m)
    g, ∂g = reduce_sol_pair_objective_gradient(p,sol,m)
    sol=nothing
    finalize(sol)
    return g, ∂g
end

function GOAT_objective_remote!(x::Vector, grad::Vector, p, tol,alg)
    a = p.params
    a .= x
    M = length(a)
    
    @everywhere f = (m) -> get_pair_objective_gradient($p,$tol,$alg,m)
    
    f = (m) -> get_pair_objective_gradient(p,tol,alg,m)
    
    
    results = pmap(f,1:M)
    
    g = results[end][1]
    ∂gs = last.(results)
    grad .= ∂gs
    return g
end

end