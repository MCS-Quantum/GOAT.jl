module GOAT

using DifferentialEquations, SparseArrays, LinearAlgebra, Distributed, Optim

export SE_action, GOAT_action, ControllableSystem, make_SE_update_function
export make_GOAT_update_function, solve_SE, solve_GOAT_eoms, make_GOAT_initial_state
export QOCProblem, GOAT_infidelity_reduce_map, SE_infidelity_reduce_map
export solve_GOAT_eoms_reduce, parallel_GOAT_fg!, find_optimal_controls, evaluate_infidelity


include("ObjectiveFunctions.jl")
export g_sm, ∂g_sm, h_sm, ∂h_sm

include("Ansatze.jl")
export window, S, general_logistic, dSdx, LO, gaussian_kernel, fourier_ansatz, derivative_fourier_ansatz, derivative_gaussian_ansatz
export gaussian_ansatz, sinusoid_kernel, morlet_kernel, morlet_ansatz, derivative_morlet_ansatz, carrier_fourier_ansatz, derivative_carrier_fourier_ansatz
export fourier_coefficient, ∂fourier_coefficient, ∂gaussian_coefficient, gaussian_coefficient, poly_coefficient, ∂poly_coefficient

include("Utilities.jl")
export make_fock_projector, direct_sum, make_operator_basis, sparse_direct_sum, isunitary
export save_opt_results, embed_square_matrix, Givens_rmul!, Givens_SUn!, SUnSUn!
export embed_A_into_B!, create_initial_vector_U_∂U, create_initial_vector_U, unpack_u_∂u
export unpack_u, unpack_us_∂us, get_sinusoidal_coefficients_from_FFT, truncated_inv_fft
export colored_noise, time_domain_signal


function SE_action(du,u,p,t,d_ms,d_ls,d_vs,c_ms,c_ls,c_vs,c_func)
    d = size(u,2) # Dimension of unitary/Hamiltonian
    lmul!(0.0,du)
    num_basis_ops = size(c_ms,1)
    for n in 1:d
        for (m,l,v) in zip(d_ms,d_ls,d_vs)
            du[l,n] += v*u[m,n]
        end
        
        for i in 1:num_basis_ops
            c_ls_ = c_ls[i]
            c_ms_ = c_ms[i]
            c_vs_ = c_vs[i]
            c = c_func(p,t,i)
            for (m,l,v) in zip(c_ms_,c_ls_,c_vs_)
                du[l,n] += c*v*u[m,n]
            end
        end     
    end
    lmul!(-im, du)
end

function SE_action(du,u,p,t,d_ms,d_ls,d_vs,c_ms,c_ls,c_vs,c_func, A::Diagonal, B::Diagonal)
    d = size(u,2) # Dimension of unitary/Hamiltonian
    lmul!(0.0,du)
    num_basis_ops = size(c_ms,1)
    for i in 1:d
        B[i,i] = exp(-im*t*A[i,i])
    end
    
    for n in 1:d
        for (m,l,v) in zip(d_ms,d_ls,d_vs)
            Bll = B[l,l]
            Bmm = conj(B[m,m])
            umn = u[m,n]
            Alm = A[l,m]
            du[l,n] += v*umn*Bll*Bmm + Alm*umn
        end
        
        for i in 1:num_basis_ops
            c_ls_ = c_ls[i]
            c_ms_ = c_ms[i]
            c_vs_ = c_vs[i]
            c = c_func(p,t,i)
            for (m,l,v) in zip(c_ms_,c_ls_,c_vs_)
                Bll = B[l,l]
                Bmm = conj(B[m,m])
                umn = u[m,n]
                du[l,n] += c*v*u[m,n]*Bll*Bmm
            end
        end     
    end
    lmul!(-im, du)
end

function GOAT_action(du, u, p, t, d_ms, d_ls, d_vs, c_ms,c_ls,c_vs, opt_param_inds, c_func::Function, ∂c_func::Function)
    d = size(u,2) # Dimension of unitary/Hamiltonian
    lmul!(0.0,du)
    num_basis_ops = size(c_ms,1)
    num_params = size(opt_param_inds,1)
    for n in 1:d
        for (m,l,v) in zip(d_ms,d_ls,d_vs)
            du[l,n] += v*u[m,n]
            for j in 1:num_params
                lj = j*d+l
                mj = j*d+m
                du[lj,n] += v*u[mj,n]
            end
        end
        
        for i in 1:num_basis_ops
            c_ls_ = c_ls[i]
            c_ms_ = c_ms[i]
            c_vs_ = c_vs[i]
            c = c_func(p,t,i)
            for (m,l,v) in zip(c_ms_,c_ls_,c_vs_)
                umn = u[m,n]
                du[l,n] += c*v*umn
                for (j,k) in enumerate(opt_param_inds)
                    lj = j*d+l
                    mj = j*d+m
                    du[lj,n] += c*v*u[mj,n]
                    dcdk = ∂c_func(p,t,i,k)
                    du[lj,n] += dcdk*v*umn
                end
            end
        end     
    end
    lmul!(-im, du)
end

function GOAT_action(du, u, p, t, d_ms, d_ls, d_vs, c_ms,c_ls,c_vs, opt_param_inds, c_func::Function, ∂c_func::Function, A::Diagonal, B::Diagonal)
    d = size(u,2) # Dimension of unitary/Hamiltonian
    lmul!(0.0,du)
    num_basis_ops = size(c_ms,1)
    num_params = size(opt_param_inds,1)
    
    for i in 1:d
        B[i,i] = exp(-im*t*A[i,i])
    end
    
    for n in 1:d      
        
        for (m,l,v) in zip(d_ms,d_ls,d_vs)
            Bll = B[l,l]
            Bmm = conj(B[m,m])
            umn = u[m,n]
            Alm = A[l,m]
            du[l,n] += v*umn*Bll*Bmm + Alm*umn
            for j in 1:num_params
                lj = j*d+l
                mj = j*d+m
                umjn = u[mj,n]
                du[lj,n] += v*umjn*Bll*Bmm+Alm*umjn
            end
        end
        
        for i in 1:num_basis_ops
            c_ls_ = c_ls[i]
            c_ms_ = c_ms[i]
            c_vs_ = c_vs[i]
            c = c_func(p,t,i)
            for (m,l,v) in zip(c_ms_,c_ls_,c_vs_)
                Bll = B[l,l]
                Bmm = conj(B[m,m])
                umn = u[m,n]
                du[l,n] += c*v*umn*Bll*Bmm
                for (j,k) in enumerate(opt_param_inds)
                    lj = j*d+l
                    mj = j*d+m
                    du[lj,n] += c*v*u[mj,n]*Bll*Bmm
                    dcdk = ∂c_func(p,t,i,k)
                    du[lj,n] += dcdk*v*umn*Bll*Bmm
                end
            end
        end     
    end
    
    lmul!(-im, du)
end

struct ControllableSystem{T,F}
    d_ms::Vector{Int64}
    d_ls::Vector{Int64}
    d_vs::Vector{ComplexF64}
    c_ls::Vector{Vector{Int64}}
    c_ms::Vector{Vector{Int64}}
    c_vs::Vector{Vector{ComplexF64}}
    coefficient_func::T
    ∂coefficient_func::F
    rotating_frame_generator::Diagonal{ComplexF64, SparseVector{ComplexF64, Int64}}
    rotating_frame_storage::Diagonal{ComplexF64, SparseVector{ComplexF64, Int64}}
    use_rotating_frame::Bool
    dim::Int64
end

function ControllableSystem(drift_op, basis_ops, c_func, ∂c_func; rotating_frame_generator=nothing)
    d_ls,d_ms,d_vs = findnz(drift_op)
    d = size(drift_op,1)
    c_ls = [findnz(op)[1] for op in basis_ops]
    c_ms = [findnz(op)[2] for op in basis_ops]
    c_vs = [findnz(op)[3] for op in basis_ops]
    if rotating_frame_generator === nothing
        RF_gen = 0*Diagonal(drift_op)
        RF_storage = similar(RF_gen)
        return ControllableSystem{typeof(c_func),typeof(∂c_func)}(d_ms, d_ls, d_vs, c_ls, c_ms, c_vs, c_func, ∂c_func, RF_gen, RF_storage, false, d)
    else
        return ControllableSystem{typeof(c_func),typeof(∂c_func)}(d_ms, d_ls, d_vs, c_ls, c_ms, c_vs, c_func, ∂c_func, rotating_frame_generator, similar(rotating_frame_generator), true , d)
    end
end

function make_SE_update_function(sys::ControllableSystem)
    if sys.use_rotating_frame
        return (du,u,p,t)-> SE_action(du, u, p, t, sys.d_ms, sys.d_ls, sys.d_vs, sys.c_ms, sys.c_ls, sys.c_vs, sys.coefficient_func, sys.rotating_frame_generator, sys.rotating_frame_storage)
    else
        return (du,u,p,t)-> SE_action(du, u, p, t, sys.d_ms, sys.d_ls, sys.d_vs, sys.c_ms, sys.c_ls, sys.c_vs, sys.coefficient_func)
    end
end

function make_GOAT_update_function(sys::ControllableSystem, opt_param_inds::Vector{Int})
    if sys.use_rotating_frame
        return (du,u,p,t)-> GOAT_action(du, u, p, t, sys.d_ms, sys.d_ls, sys.d_vs, sys.c_ms, sys.c_ls, sys.c_vs, opt_param_inds, sys.coefficient_func, sys.∂coefficient_func, sys.rotating_frame_generator, sys.rotating_frame_storage)
    else
        return (du,u,p,t)-> GOAT_action(du, u, p, t, sys.d_ms, sys.d_ls, sys.d_vs, sys.c_ms, sys.c_ls, sys.c_vs, opt_param_inds, sys.coefficient_func, sys.∂coefficient_func)
    end
end

function make_GOAT_initial_state(d,opt_param_inds)
    n_us = size(opt_param_inds,1)+1
    u0 = zeros(ComplexF64,d*n_us,d)
    for n in 1:d
        u0[n,n] = 1.0
    end
    return u0
end

function solve_SE(sys::ControllableSystem, Tmax::Float64, p::Vector{Float64}; args...)
    tspan = (0.0, Tmax)
    f = make_SE_update_function(sys)
    u0 = Matrix{ComplexF64}(I, sys.dim, sys.dim)
    prob = ODEProblem(f,u0, tspan, p)
    sol = solve(prob; args...)
    return sol
end


function solve_GOAT_eoms(sys::ControllableSystem, opt_param_inds::Vector{Int}, Tmax::Float64, p::Vector{Float64} ; args...)
    tspan = (0.0,Tmax)
    g = make_GOAT_update_function(sys, opt_param_inds)
    u0 = make_GOAT_initial_state(sys.dim, opt_param_inds)
    prob = ODEProblem(g,u0,tspan,p)
    sol = solve(prob; args...)
    return sol
end

struct QOCProblem
    Pc::Array{ComplexF64}
    Pc_dim::Int64
    Pa::Array{ComplexF64}
    Pa_dim::Int64
    target::Array{ComplexF64}
    control_time::Float64
end

QOCProblem(target, control_time, Pc, Pa) = QOCProblem(Pc, Int(tr(Pc)), Pa, Int(tr(Pa)), target, control_time)

function GOAT_infidelity_reduce_map(sys::ControllableSystem,prob::QOCProblem,goat_sol)
    d = sys.dim    
    Pc = prob.Pc
    Pc_dim = prob.Pc_dim
    target = prob.target
    goatU = goat_sol.u[end]
    n_params = size(goatU,1)÷d
    Ut = goatU[1:d,:]
    g = g_sm(Pc*target*Pc, Ut; dim=Pc_dim)

    ∂g_vec = Float64[]
    for i in 1:n_params-1
        ∂Ut = goatU[i*d+1:(i+1)*d,:]
        ∂g = ∂g_sm(Pc*target*Pc, Ut, ∂Ut ; dim=Pc_dim)
        push!(∂g_vec,∂g)
    end
    return g, ∂g_vec
end

function SE_infidelity_reduce_map(sys::ControllableSystem,prob::QOCProblem,SE_sol)
    d = sys.dim
    Pc = prob.Pc
    Pc_dim = prob.Pc_dim
    target = prob.target
    Ut = SE_sol.u[end]
    g = g_sm(Pc*target*Pc, Ut; dim=Pc_dim)
    return g
end

function solve_GOAT_eoms_reduce(x, sys::ControllableSystem, prob::QOCProblem, opt_param_inds, GOAT_reduce_map, diffeq_options)
    T = prob.control_time
    goat_sol = solve_GOAT_eoms(sys,opt_param_inds,T,x; diffeq_options...)
    out = GOAT_reduce_map(sys, prob, goat_sol)
    g = first(out)
    ∂gs = last(out)
    return g, ∂gs
end

function parallel_GOAT_fg!(F, G, x, sys::ControllableSystem, prob::QOCProblem, SE_reduce_map, GOAT_reduce_map, diffeq_options; num_params_per_GOAT=nothing)
    T = prob.control_time
    if G !== nothing
        num_params = size(x,1)
        if num_params_per_GOAT === nothing
            num_params_per_GOAT = num_params
        end
        goat_param_indices = collect.(collect(Iterators.partition(1:num_params, num_params_per_GOAT)))
        f = y -> solve_GOAT_eoms_reduce(x, sys, prob, y, GOAT_reduce_map, diffeq_options)
        out = pmap(f,goat_param_indices)
        gs = first.(out)
        # @assert gs[1] ≈ gs[end] # Trivial sanity check
        for (i,inds) in enumerate(goat_param_indices)
            ∂gs = last(out[i])
            G[inds] .= ∂gs
        end
        g = gs[1]
    else
        sol = solve_SE(sys,T,x; diffeq_options...)
        g = SE_reduce_map(sys,prob,sol)
    end
    
    if F !== nothing
        return g
    end

end

function parallel_GOAT_fg!(F, G, x, p_storage, opt_param_inds, sys::ControllableSystem, prob::QOCProblem, SE_reduce_map, GOAT_reduce_map, diffeq_options; num_params_per_GOAT=nothing)
    T = prob.control_time
    p_storage[opt_param_inds] .= x # Update the storage vector with new parameters from optimization
    if G !== nothing
        num_params = size(x,1)
        if num_params_per_GOAT === nothing
            num_params_per_GOAT = num_params
        end
        goat_param_indices = collect.(collect(Iterators.partition(opt_param_inds, num_params_per_GOAT)))
        f = y -> solve_GOAT_eoms_reduce(p_storage, sys, prob, y, GOAT_reduce_map, diffeq_options)
        out = pmap(f,goat_param_indices)
        gs = first.(out)
        # @assert gs[1] ≈ gs[end] # Trivial sanity check
        for (i,inds) in enumerate(goat_param_indices)
            ∂gs = last(out[i])
            G[inds] .= ∂gs
        end
        g = gs[1]
    else
        sol = solve_SE(sys,T,p_storage; diffeq_options...)
        g = SE_reduce_map(sys,prob,sol)
    end
    
    if F !== nothing
        return g
    end

end

function find_optimal_controls(p0, sys::ControllableSystem, prob::QOCProblem, SE_reduce_map, GOAT_reduce_map, diffeq_options, optim_alg, optim_options ; num_params_per_GOAT=nothing)
    fg!(F,G,x) = parallel_GOAT_fg!(F,G,x,sys, prob, SE_reduce_map, GOAT_reduce_map, diffeq_options; num_params_per_GOAT=num_params_per_GOAT)
    res = Optim.optimize(Optim.only_fg!(fg!), p0, optim_alg, optim_options)
    return res
end
    
function find_optimal_controls(p0, opt_param_inds, sys::ControllableSystem, prob::QOCProblem, SE_reduce_map, GOAT_reduce_map, diffeq_options, optim_alg, optim_options ; num_params_per_GOAT=nothing)
    p_storage = deepcopy(p0)
    fg!(F,G,x) = parallel_GOAT_fg!(F, G, x, p_storage, opt_param_inds, sys, prob, SE_reduce_map, GOAT_reduce_map, diffeq_options; num_params_per_GOAT=num_params_per_GOAT)
    x0 = @view p0[opt_param_inds]
    res = Optim.optimize(Optim.only_fg!(fg!), x0, optim_alg, optim_options)
    return res
end

function evaluate_infidelity(p0::Vector{Float64}, sys::ControllableSystem, prob::QOCProblem, SE_reduce_map, diffeq_options)
    T = prob.control_time
    sol = solve_SE(sys,T,p0; diffeq_options...)
    g = SE_reduce_map(sys, prob, sol)
    return g
end

function evaluate_infidelity(ps::Vector{Vector{Float64}}, sys::ControllableSystem, prob::QOCProblem, SE_reduce_map)

    f = y -> evaluate_infidelity(y, sys, prob, SE_reduce_map, diffeq_options)
    out = pmap(f,ps)
    return out
end

end # module
