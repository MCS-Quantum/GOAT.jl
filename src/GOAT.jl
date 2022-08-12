module GOAT

using DifferentialEquations, SparseArrays, LinearAlgebra

export SE_action, GOAT_action, ControllableSystem, make_SE_update_function, make_GOAT_update_function, solve_SE, solve_GOAT_eoms, make_GOAT_initial_state

include("ObjectiveFunctions.jl")
export g_sm, ∂g_sm, h_sm, ∂h_sm

include("Ansatze.jl")
export window, S, general_logistic, dSdx, LO, gaussian_kernel, fourier_ansatz, derivative_fourier_ansatz, derivative_gaussian_ansatz, gaussian_ansatz, sinusoid_kernel, morlet_kernel, morlet_ansatz, derivative_morlet_ansatz, carrier_fourier_ansatz, derivative_carrier_fourier_ansatz, fourier_coefficient, ∂fourier_coefficient, ∂gaussian_coefficient, gaussian_coefficient

include("Utilities.jl")
export make_fock_projector, direct_sum, make_operator_basis, sparse_direct_sum, isunitary, save_opt_results, embed_square_matrix, Givens_rmul!, Givens_SUn!, SUnSUn!, embed_A_into_B!, create_initial_vector_U_∂U, create_initial_vector_U, unpack_u_∂u, unpack_u, unpack_us_∂us, get_sinusoidal_coefficients_from_FFT, truncated_inv_fft


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

function GOAT_action(du, u, p, t, d_ms, d_ls, d_vs, c_ms,c_ls,c_vs, param_inds, c_func::Function, ∂c_func::Function)
    d = size(u,2) # Dimension of unitary/Hamiltonian
    lmul!(0.0,du)
    num_basis_ops = size(c_ms,1)
    num_params = size(param_inds,1)
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
                for (j,k) in enumerate(param_inds)
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

function GOAT_action(du, u, p, t, d_ms, d_ls, d_vs, c_ms,c_ls,c_vs, param_inds, c_func::Function, ∂c_func::Function, A::Diagonal, B::Diagonal)
    d = size(u,2) # Dimension of unitary/Hamiltonian
    lmul!(0.0,du)
    num_basis_ops = size(c_ms,1)
    num_params = size(param_inds,1)
    
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
            for n in eachindex(c_ls_)
                m = c_ms_[n]
                l = c_ls_[n]
                v = c_vs_[n]
                Bll = B[l,l]
                Bmm = conj(B[m,m])
                umn = u[m,n]
                du[l,n] += c*v*umn*Bll*Bmm
                for (j,k) in enumerate(param_inds)
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

function make_GOAT_update_function(sys::ControllableSystem, param_inds::Vector{Int})
    if sys.use_rotating_frame
        return (du,u,p,t)-> GOAT_action(du, u, p, t, sys.d_ms, sys.d_ls, sys.d_vs, sys.c_ms, sys.c_ls, sys.c_vs, param_inds, sys.coefficient_func, sys.∂coefficient_func, sys.rotating_frame_generator, sys.rotating_frame_storage)
    else
        return (du,u,p,t)-> GOAT_action(du, u, p, t, sys.d_ms, sys.d_ls, sys.d_vs, sys.c_ms, sys.c_ls, sys.c_vs, param_inds, sys.coefficient_func, sys.∂coefficient_func)
    end
end

function make_GOAT_initial_state(d,param_inds)
    n_us = size(param_inds,1)+1
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


function solve_GOAT_eoms(sys::ControllableSystem, param_inds::Vector{Int}, Tmax::Float64, p::Vector{Float64} ; args...)
    tspan = (0.0,Tmax)
    g = make_GOAT_update_function(sys, param_inds)
    u0 = make_GOAT_initial_state(sys.dim, param_inds)
    prob = ODEProblem(g,u0,tspan,p)
    sol = solve(prob; args...)
    return sol
end


end # module
