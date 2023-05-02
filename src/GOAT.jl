# Copyright © 2023, UChicago Argonne, LLC
# All Rights Reserved
# Software Name: GOAT.jl
# By: Argonne National Laboratory
# OPEN-SOURCE LICENSE

# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

# ******************************************************************************************************
# DISCLAIMER

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, 
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE 
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, 
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR 
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE 
# USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ***************************************************************************************************


module GOAT

using OrdinaryDiffEq, SparseArrays, LinearAlgebra, Distributed, Optim

export SE_action, GOAT_action, ControllableSystem, make_SE_update_function
export make_GOAT_update_function, solve_SE, solve_GOAT_eoms, make_GOAT_initial_state
export QOCProblem, GOAT_infidelity_reduce_map, SE_infidelity_reduce_map
export QOCParameters
export solve_GOAT_eoms_reduce, parallel_GOAT_fg!, find_optimal_controls, evaluate_objective


include("ObjectiveFunctions.jl")
export g_sm, ∂g_sm, h_sm, ∂h_sm

include("Ansatze.jl")
export fourier_coefficient,
    ∂fourier_coefficient,
    ∂gaussian_coefficient,
    gaussian_coefficient,
    poly_coefficient,
    ∂poly_coefficient

include("Utilities.jl")
export get_sinusoidal_basis_parameters, time_domain_signal
export window, S, general_logistic, dSdx, LO, gaussian_kernel, flat_top_cosine
export sinusoid_kernel, morlet_kernel
export colored_noise, test_derivatives



"""
Instantiate a ControllableSystem struct.
"""
struct ControllableSystem{A,B,C,D,E}
    d_ms::A
    d_ls::A
    d_vs::B
    c_ls::Vector{Vector{Int64}}
    c_ms::Vector{Vector{Int64}}
    c_vs::Vector{Vector{ComplexF64}}
    coefficient_func::C
    ∂coefficient_func::D
    reference_frame_generator::E
    reference_frame_storage::E
    use_rotating_frame::Bool
    dim::Int64
    orthogonal_basis::Bool
    tol::Float64
end

"""
    ControllableSystem(drift_op, basis_ops, c_func, ∂c_func)

Instantiate a ControllableSystem struct from input operators and functions.

The `basis_ops` is a vector of basis operators which will be weighted by the
coefficient functions `c_func`. `∂c_func` computes the first derivative of 
c_func evaluated at an input.
"""
function ControllableSystem(drift_op, basis_ops, c_func, ∂c_func)
    d_ls, d_ms, d_vs = findnz(drift_op)
    d = size(drift_op, 1)
    c_ls = [findnz(op)[1] for op in basis_ops]
    c_ms = [findnz(op)[2] for op in basis_ops]
    c_vs = [findnz(op)[3] for op in basis_ops]
    return ControllableSystem{
        typeof(d_ls),
        typeof(d_vs),
        typeof(c_func),
        typeof(∂c_func),
        Nothing,
    }(
        d_ms,
        d_ls,
        d_vs,
        c_ls,
        c_ms,
        c_vs,
        c_func,
        ∂c_func,
        nothing,
        nothing,
        false,
        d,
        false,
        0.0,
    )

end

"""
    NoDiffControllableSystem(drift_op, basis_ops, c_func)

Instantiate a ControllableSystem struct with no specified `∂c_func`.

The `basis_ops` is a vector of basis operators which will be weighted by the
coefficient functions `c_func`.
"""
function NoDiffControllableSystem(drift_op, basis_ops, c_func)
    ∂c_func = nothing
    d_ls, d_ms, d_vs = findnz(drift_op)
    d = size(drift_op, 1)
    c_ls = [findnz(op)[1] for op in basis_ops]
    c_ms = [findnz(op)[2] for op in basis_ops]
    c_vs = [findnz(op)[3] for op in basis_ops]
    return ControllableSystem{
        typeof(d_ls),
        typeof(d_vs),
        typeof(c_func),
        typeof(∂c_func),
        Nothing,
    }(
        d_ms,
        d_ls,
        d_vs,
        c_ls,
        c_ms,
        c_vs,
        c_func,
        ∂c_func,
        nothing,
        nothing,
        false,
        d,
        false,
        0.0,
    )

end

"""
    ControllableSystem(drift_op, basis_ops, RF_generator::Eigen, c_func, ∂c_func; <keyword arguments>)

Instantiate a ControllableSystem struct with a specified `RF_generator`. 

The `RF_generator` is a specification of the time-independent generator of the rotating
reference frame. i.e., if ``∂V(t)=A*V(t)`` where A is the generator of type ``Eigen``.

# Keyword Arguments
- `sparse_tol::Float`: A tolerance which defines a threshold to discard matrix elements
"""
function ControllableSystem(
    drift_op,
    basis_ops,
    RF_generator::Eigen,
    c_func,
    ∂c_func;
    sparse_tol = 1e-12,
)
    d = size(drift_op, 1)
    F = RF_generator
    N = size(basis_ops, 1)
    c_ls = []
    c_ms = []
    c_vs = []
    as = F.values
    a_diffs = zeros(ComplexF64, d, d)
    aj_drift_aks = zeros(ComplexF64, d, d)
    aj_hi_aks = zeros(ComplexF64, d, N, d)
    for j = 1:d
        for k = 1:d
            aj = F.values[j]
            ak = F.values[k]
            a_diffs[j, k] = aj - ak
            aj_vec = @view F.vectors[:, j]
            ak_vec = @view F.vectors[:, k]
            aj_drift_ak = adjoint(aj_vec) * drift_op * ak_vec
            aj_drift_aks[j, k] = aj_drift_ak

            new_basis_op = sparse(aj_vec * adjoint(ak_vec))
            droptol!(new_basis_op, sparse_tol)
            for i = 1:N
                aj_hi_ak = adjoint(aj_vec) * basis_ops[i] * ak_vec
                aj_hi_aks[j, i, k] = aj_hi_ak
            end

            ls, ms, vs = findnz(new_basis_op)
            push!(c_ls, ls)
            push!(c_ms, ms)
            push!(c_vs, vs)
        end
    end

    function new_c_func(p, t, j, k)
        c = 0.0 + 0.0im
        diag_term = 0.0 + 0.0im
        if j == k
            diag_term = as[j]
        end
        for i = 1:N
            c += c_func(p, t, i) * aj_hi_aks[j, i, k]
        end
        adiff = a_diffs[j, k]
        aj_drift_ak = aj_drift_aks[j, k]
        return cis(t * adiff) * (aj_drift_ak + c + diag_term)
    end

    function new_∂c_func(p, t, j, k, m)
        c = 0.0 + 0.0im
        for i = 1:N
            c += ∂c_func(p, t, i, m) * aj_hi_aks[j, i, k]
        end
        adiff = a_diffs[j, k]
        return cis(t * adiff) * c
    end

    return ControllableSystem{
        Nothing,
        Nothing,
        typeof(new_c_func),
        typeof(new_∂c_func),
        Nothing,
    }(
        nothing,
        nothing,
        nothing,
        c_ls,
        c_ms,
        c_vs,
        new_c_func,
        new_∂c_func,
        nothing,
        nothing,
        false,
        d,
        true,
        sparse_tol,
    )
end

"""
    ControllableSystem(drift_op, basis_ops, RF_generator::Matrix, c_func, ∂c_func; <keyword arguments>)

Instantiate a ControllableSystem struct with a specified `RF_generator`. 

The `RF_generator` is a specification of the time-independent generator of the rotating
reference frame. i.e., if ``∂V(t)=A*V(t)`` where A is the generator of type ``Matrix``.

# Keyword Arguments
- `sparse_tol::Float`: A tolerance which defines a threshold to discard matrix elements
"""
function ControllableSystem(
    drift_op,
    basis_ops,
    RF_generator::Matrix,
    c_func,
    ∂c_func;
    sparse_tol = 1e-12,
)
    d = size(drift_op, 1)
    F = eigen(RF_generator)
    N = size(basis_ops, 1)
    c_ls = []
    c_ms = []
    c_vs = []
    as = F.values
    a_diffs = zeros(ComplexF64, d, d)
    aj_drift_aks = zeros(ComplexF64, d, d)
    aj_hi_aks = zeros(ComplexF64, d, N, d)
    for j = 1:d
        for k = 1:d
            aj = F.values[j]
            ak = F.values[k]
            a_diffs[j, k] = aj - ak
            aj_vec = @view F.vectors[:, j]
            ak_vec = @view F.vectors[:, k]
            aj_drift_ak = adjoint(aj_vec) * drift_op * ak_vec
            aj_drift_aks[j, k] = aj_drift_ak

            new_basis_op = sparse(aj_vec * adjoint(ak_vec))
            droptol!(new_basis_op, sparse_tol)
            for i = 1:N
                aj_hi_ak = adjoint(aj_vec) * basis_ops[i] * ak_vec
                aj_hi_aks[j, i, k] = aj_hi_ak
            end

            ls, ms, vs = findnz(new_basis_op)
            push!(c_ls, ls)
            push!(c_ms, ms)
            push!(c_vs, vs)
        end
    end

    function new_c_func(p, t, j, k)
        c = 0.0 + 0.0im
        diag_term = 0.0 + 0.0im
        if j == k
            diag_term = as[j]
        end
        for i = 1:N
            c += c_func(p, t, i) * aj_hi_aks[j, i, k]
        end
        adiff = a_diffs[j, k]
        aj_drift_ak = aj_drift_aks[j, k]
        return cis(t * adiff) * (aj_drift_ak + c + diag_term)
    end

    function new_∂c_func(p, t, j, k, m)
        c = 0.0 + 0.0im
        for i = 1:N
            c += ∂c_func(p, t, i, m) * aj_hi_aks[j, i, k]
        end
        adiff = a_diffs[j, k]
        return cis(t * adiff) * c
    end

    return ControllableSystem{
        Nothing,
        Nothing,
        typeof(new_c_func),
        typeof(new_∂c_func),
        Nothing,
    }(
        nothing,
        nothing,
        nothing,
        c_ls,
        c_ms,
        c_vs,
        new_c_func,
        new_∂c_func,
        nothing,
        nothing,
        false,
        d,
        true,
        sparse_tol,
    )
end

"""
    ControllableSystem(drift_op, basis_ops, RF_generator::Eigen, c_func, ∂c_func; <keyword arguments>)

Instantiate a ControllableSystem struct with a specified `RF_generator`. 

The `RF_generator` is a specification of the time-independent generator of the rotating
reference frame. i.e., if ``∂V(t)=A*V(t)`` where A is the generator of type ``LinearAlgebra.Diagonal``.

# Keyword Arguments
- `sparse_tol::Float`: A tolerance which defines a threshold to discard matrix elements
"""
function ControllableSystem(
    drift_op,
    basis_ops,
    RF_generator::LinearAlgebra.Diagonal,
    c_func,
    ∂c_func,
)
    d_ls, d_ms, d_vs = findnz(drift_op)
    d = size(drift_op, 1)
    c_ls = [findnz(op)[1] for op in basis_ops]
    c_ms = [findnz(op)[2] for op in basis_ops]
    c_vs = [findnz(op)[3] for op in basis_ops]
    return ControllableSystem{
        typeof(d_ls),
        typeof(d_vs),
        typeof(c_func),
        typeof(∂c_func),
        typeof(RF_generator),
    }(
        d_ms,
        d_ls,
        d_vs,
        c_ls,
        c_ms,
        c_vs,
        c_func,
        ∂c_func,
        RF_generator,
        similar(RF_generator),
        true,
        d,
        false,
        0.0,
    )
end

"""
Instantiate a QOCProblem struct.
"""
struct QOCProblem
    Pc::Array{ComplexF64}
    Pc_dim::Int64
    Pa::Array{ComplexF64}
    Pa_dim::Int64
    target::Array{ComplexF64}
    control_time::Float64
end

"""
    QOCProblem(target, control_time, Pc, Pa)

Instantiate a QOCProblem struct with specified input operators and control time.

# Arguments
- `target::Array{ComplexF64}`: the target unitary operator in the computational subspace.
- `control_time::Float64`: The duration of the control.
- `Pc::Array{ComplexF64}`: A projector from the full unitary operator on the system to the computational subspace
- `Pa::Array{ComplexF64}`: A projector from the full unitary operator on the system to any ancillary subspace
"""
QOCProblem(target, control_time, Pc, Pa) =
    QOCProblem(Pc, Int(round(tr(Pc))), Pa, Int(round(tr(Pa))), target, control_time)


"""
Instantiate a QOCParameters struct.
"""
struct QOCParameters{A,B,C,D,E}
    ODE_options::NamedTuple
    SE_reduce_map::A
    GOAT_reduce_map::B
    optim_alg::C
    optim_options::D
    num_params_per_GOAT::E
end

"""
    QOCParameters(ODE_options,SE_reduce_map, GOAT_reduce_map, optim_alg, optim_options; <keyword arguments> )

Instantiate a QOCParameters struct with specified parameters. 

# Arguments
- `ODE_options::NamedTuple`: The settings that will be input to the ODE solver.
- `SE_reduce_map::Function`: A function mapping the output from the Schrodinger equation to the objective value.
- `GOAT_reduce_map::Function`: A function mapping the output from the GOAT E.O.M.s to the objective value and gradient.
- `optim_alg::Optim Algorithm`: The specific optimization algorithm specified via Optim.jl
- `optim_options::Optim options`: The optimization algorithm options specified via Optim.jl
"""
function QOCParameters(
    ODE_options,
    SE_reduce_map,
    GOAT_reduce_map,
    optim_alg,
    optim_options;
    num_params_per_GOAT = nothing,
)
    return QOCParameters{
        typeof(SE_reduce_map),
        typeof(GOAT_reduce_map),
        typeof(optim_alg),
        typeof(optim_options),
        typeof(num_params_per_GOAT),
    }(
        ODE_options,
        SE_reduce_map,
        GOAT_reduce_map,
        optim_alg,
        optim_options,
        num_params_per_GOAT,
    )
end

function SE_action!(
    du::Array{ComplexF64},
    u::Array{ComplexF64},
    p::Vector{Float64},
    t::Float64,
    c_ms::Vector{Vector{Int64}},
    c_ls::Vector{Vector{Int64}},
    c_vs::Vector{Vector{ComplexF64}},
    c_func::Function,
)
    d = size(u, 2) # Dimension of unitary/Hamiltonian
    lmul!(0.0, du)
    num_basis_ops = size(c_ms, 1)
    for i = 1:num_basis_ops
        c_ls_ = c_ls[i]
        c_ms_ = c_ms[i]
        c_vs_ = c_vs[i]
        c = c_func(p, t, i)
        for (m, l, v) in zip(c_ms_, c_ls_, c_vs_)
            for n = 1:d
                du[l, n] += c * v * u[m, n]
            end
        end
    end
    lmul!(-im, du)
end

function SE_action!(
    du::Array{ComplexF64},
    u::Array{ComplexF64},
    p::Vector{Float64},
    t::Float64,
    c_ms::Vector{Vector{Int64}},
    c_ls::Vector{Vector{Int64}},
    c_vs::Vector{Vector{ComplexF64}},
    c_func::Function,
    linear_index::LinearIndices,
    tol::Float64,
)
    d = size(u, 2) # Dimension of unitary/Hamiltonian
    lmul!(0.0, du)
    for j = 1:d
        k = 1
        while k <= j
            cjk = c_func(p, t, j, k)
            if abs2(cjk) <= tol
                k += 1
                continue
            end
            ckj = conj(cjk)
            q = linear_index[j, k]
            r = linear_index[k, j]
            c_ls_jk = c_ls[q]
            c_ms_jk = c_ms[q]
            c_vs_jk = c_vs[q]
            c_ls_kj = c_ls[r]
            c_ms_kj = c_ms[r]
            c_vs_kj = c_vs[r]
            for (m1, l1, v1, m2, l2, v2) in
                zip(c_ms_jk, c_ls_jk, c_vs_jk, c_ms_kj, c_ls_kj, c_vs_kj)
                for n = 1:d
                    du[l1, n] += cjk * v1 * u[m1, n]
                    du[l2, n] += ckj * v2 * u[m2, n]
                end
            end
            k += 1
        end
    end
    lmul!(-im, du)
end

"""
    SE_action!(du, u, p, t, d_ms, d_ls, d_vs, c_ms, c_ls, c_vs, c_func)

Compute the action of the Schrodinger equation on a matrix `u` and place in `du`.

# Arguments
- `du::Array{ComplexF64}`: The array for the derivative `t`
- `u::Array{ComplexF64}`: The array for the state at time `t`
- `p::Vector{Float64}`: The parameter vector defining the evolution. 
- `t::Float64`: The time.
- `d_ms::Vector{Vector{Int64}}`: The sparse representation of the drift operator's first index.
- `d_ls::Vector{Vector{Int64}}`: The sparse representation of the drift operators's second index.
- `d_vs::Vector{Vector{ComplexF64}}`: The sparse representation of the mtrix element of the drift operator.
- `c_ms::Vector{Vector{Int64}}`: The sparse representation of the control operators' first index.
- `c_ls::Vector{Vector{Int64}}`: The sparse representation of the control operators' second index.
- `c_vs::Vector{Vector{ComplexF64}}`: The sparse representation of the mtrix element of the control operators.
- `c_func::Function`: The function that computes the time-dependent coefficients of the control operators.
"""
function SE_action!(
    du::Array{ComplexF64},
    u::Array{ComplexF64},
    p::Vector{Float64},
    t::Float64,
    d_ms::Vector{Int64},
    d_ls::Vector{Int64},
    d_vs::Vector{ComplexF64},
    c_ms::Vector{Vector{Int64}},
    c_ls::Vector{Vector{Int64}},
    c_vs::Vector{Vector{ComplexF64}},
    c_func::Function,
)
    d = size(u, 2) # Dimension of unitary/Hamiltonian
    lmul!(0.0, du)
    num_basis_ops = size(c_ms, 1)
    for n = 1:d
        for (m, l, v) in zip(d_ms, d_ls, d_vs)
            du[l, n] += v * u[m, n]
        end

        for i = 1:num_basis_ops
            c_ls_ = c_ls[i]
            c_ms_ = c_ms[i]
            c_vs_ = c_vs[i]
            c = c_func(p, t, i)
            for (m, l, v) in zip(c_ms_, c_ls_, c_vs_)
                du[l, n] += c * v * u[m, n]
            end
        end
    end
    lmul!(-im, du)
end

function SE_action!(
    du::Array{ComplexF64},
    u::Array{ComplexF64},
    p::Vector{Float64},
    t::Float64,
    d_ms::Vector{Int64},
    d_ls::Vector{Int64},
    d_vs::Vector{ComplexF64},
    c_ms::Vector{Vector{Int64}},
    c_ls::Vector{Vector{Int64}},
    c_vs::Vector{Vector{ComplexF64}},
    c_func::Function,
    A::Diagonal,
    B::Diagonal,
)
    d = size(u, 2) # Dimension of unitary/Hamiltonian
    lmul!(0.0, du)
    num_basis_ops = size(c_ms, 1)
    for i = 1:d
        B[i, i] = cis(-t * A[i, i])
    end

    for n = 1:d
        for (m, l, v) in zip(d_ms, d_ls, d_vs)
            Bll = B[l, l]
            Bmm = conj(B[m, m])
            umn = u[m, n]
            Alm = A[l, m]
            du[l, n] += v * umn * Bll * Bmm + Alm * umn
        end

        for i = 1:num_basis_ops
            c_ls_ = c_ls[i]
            c_ms_ = c_ms[i]
            c_vs_ = c_vs[i]
            c = c_func(p, t, i)
            for (m, l, v) in zip(c_ms_, c_ls_, c_vs_)
                Bll = B[l, l]
                Bmm = conj(B[m, m])
                umn = u[m, n]
                du[l, n] += c * v * umn * Bll * Bmm
            end
        end
    end
    lmul!(-im, du)
end

function GOAT_action!(
    du::Array{ComplexF64},
    u::Array{ComplexF64},
    p::Vector{Float64},
    t::Float64,
    c_ms::Vector{Vector{Int64}},
    c_ls::Vector{Vector{Int64}},
    c_vs::Vector{Vector{ComplexF64}},
    opt_param_inds::Vector{Int64},
    c_func::Function,
    ∂c_func::Function,
)
    d = size(u, 2) # Dimension of unitary/Hamiltonian
    lmul!(0.0, du)
    num_basis_ops = size(c_ms, 1)
    for i = 1:num_basis_ops
        c_ls_ = c_ls[i]
        c_ms_ = c_ms[i]
        c_vs_ = c_vs[i]
        c = c_func(p, t, i)
        for (m, l, v) in zip(c_ms_, c_ls_, c_vs_)
            for n = 1:d
                umn = u[m, n]
                du[l, n] += c * v * umn
                for (j, k) in enumerate(opt_param_inds)
                    lj = j * d + l
                    mj = j * d + m
                    du[lj, n] += c * v * u[mj, n]
                    dcdk = ∂c_func(p, t, i, k)
                    du[lj, n] += dcdk * v * umn
                end
            end
        end
    end
    lmul!(-im, du)
end

function GOAT_action!(
    du::Array{ComplexF64},
    u::Array{ComplexF64},
    p::Vector{Float64},
    t::Float64,
    c_ms::Vector{Vector{Int64}},
    c_ls::Vector{Vector{Int64}},
    c_vs::Vector{Vector{ComplexF64}},
    opt_param_inds::Vector{Int64},
    c_func::Function,
    ∂c_func::Function,
    linear_index::LinearIndices,
    tol::Float64,
)
    d = size(u, 2) # Dimension of unitary/Hamiltonian
    lmul!(0.0, du)
    for j = 1:d
        k = 1
        while k <= j
            cjk = c_func(p, t, j, k)
            if abs2(cjk) < tol
                k += 1
                continue
            end
            ckj = conj(cjk)
            q = linear_index[j, k]
            r = linear_index[k, j]
            c_ls_jk = c_ls[q]
            c_ms_jk = c_ms[q]
            c_vs_jk = c_vs[q]
            c_ls_kj = c_ls[r]
            c_ms_kj = c_ms[r]
            c_vs_kj = c_vs[r]
            for (m1, l1, v1, m2, l2, v2) in
                zip(c_ms_jk, c_ls_jk, c_vs_jk, c_ms_kj, c_ls_kj, c_vs_kj)
                for n = 1:d
                    um1n = u[m1, n]
                    um2n = u[m2, n]
                    du[l1, n] += cjk * v1 * um1n
                    du[l2, n] += ckj * v2 * um2n
                    for (j_, k_) in enumerate(opt_param_inds)
                        l1j_ = j_ * d + l1
                        m1j_ = j_ * d + m1
                        l2j_ = j_ * d + l2
                        m2j_ = j_ * d + m2
                        du[l1j_, n] += cjk * v1 * u[m1j_, n]
                        du[l2j_, n] += ckj * v2 * u[m2j_, n]
                        dcjkdk_ = ∂c_func(p, t, j, k, k_)
                        du[l1j_, n] += dcjkdk_ * v1 * um1n
                        du[l2j_, n] += dcjkdk_ * v2 * um2n
                    end
                end
            end
            k += 1
        end
    end
    lmul!(-im, du)
end

"""
    GOAT_action!(du, u, p, t, d_ms, d_ls, d_vs, c_ms, c_ls, c_vs, opt_param_inds, c_func, ∂c_func)

Compute the action of the GOAT equation of motions on a matrix `u` and place in `du`.

# Arguments
- `du::Array{ComplexF64}`: The array for the derivative `t`
- `u::Array{ComplexF64}`: The array for the state at time `t`
- `p::Vector{Float64}`: The parameter vector defining the evolution. 
- `t::Float64`: The time.
- `d_ms::Vector{Vector{Int64}}`: The sparse representation of the drift operator's first index.
- `d_ls::Vector{Vector{Int64}}`: The sparse representation of the drift operators's second index.
- `d_vs::Vector{Vector{ComplexF64}}`: The sparse representation of the mtrix element of the drift operator.
- `c_ms::Vector{Vector{Int64}}`: The sparse representation of the control operators' first index.
- `c_ls::Vector{Vector{Int64}}`: The sparse representation of the control operators' second index.
- `c_vs::Vector{Vector{ComplexF64}}`: The sparse representation of the mtrix element of the control operators.
- `opt_param_inds::Vector{Int64}`: The indices of the parameter vector `p` to propogate a derivative for
- `c_func::Function`: The function that computes the time-dependent coefficients of the control operators.
- `∂c_func::Function`: The function that computes the derivative of the time-dependent coefficients of the control operators.
"""
function GOAT_action!(
    du::Array{ComplexF64},
    u::Array{ComplexF64},
    p::Vector{Float64},
    t::Float64,
    d_ms::Vector{Int64},
    d_ls::Vector{Int64},
    d_vs::Vector{ComplexF64},
    c_ms::Vector{Vector{Int64}},
    c_ls::Vector{Vector{Int64}},
    c_vs::Vector{Vector{ComplexF64}},
    opt_param_inds::Vector{Int64},
    c_func::Function,
    ∂c_func::Function,
)
    d = size(u, 2) # Dimension of unitary/Hamiltonian
    lmul!(0.0, du)
    num_basis_ops = size(c_ms, 1)
    num_params = size(opt_param_inds, 1)
    for n = 1:d
        for (m, l, v) in zip(d_ms, d_ls, d_vs)
            du[l, n] += v * u[m, n]
            for j = 1:num_params
                lj = j * d + l
                mj = j * d + m
                du[lj, n] += v * u[mj, n]
            end
        end

        for i = 1:num_basis_ops
            c_ls_ = c_ls[i]
            c_ms_ = c_ms[i]
            c_vs_ = c_vs[i]
            c = c_func(p, t, i)
            for (m, l, v) in zip(c_ms_, c_ls_, c_vs_)
                umn = u[m, n]
                du[l, n] += c * v * umn
                for (j, k) in enumerate(opt_param_inds)
                    lj = j * d + l
                    mj = j * d + m
                    du[lj, n] += c * v * u[mj, n]
                    dcdk = ∂c_func(p, t, i, k)
                    du[lj, n] += dcdk * v * umn
                end
            end
        end
    end
    lmul!(-im, du)
end

function GOAT_action!(
    du::Array{ComplexF64},
    u::Array{ComplexF64},
    p::Vector{Float64},
    t::Float64,
    d_ms::Vector{Int64},
    d_ls::Vector{Int64},
    d_vs::Vector{ComplexF64},
    c_ms::Vector{Vector{Int64}},
    c_ls::Vector{Vector{Int64}},
    c_vs::Vector{Vector{ComplexF64}},
    opt_param_inds::Vector{Int64},
    c_func::Function,
    ∂c_func::Function,
    A::Diagonal,
    B::Diagonal,
)
    d = size(u, 2) # Dimension of unitary/Hamiltonian
    lmul!(0.0, du)
    num_basis_ops = size(c_ms, 1)
    num_params = size(opt_param_inds, 1)

    for i = 1:d
        B[i, i] = cis(-t * A[i, i])
    end

    for n = 1:d

        for (m, l, v) in zip(d_ms, d_ls, d_vs)
            Bll = B[l, l]
            Bmm = conj(B[m, m])
            umn = u[m, n]
            Alm = A[l, m]
            du[l, n] += v * umn * Bll * Bmm + Alm * umn
            for j = 1:num_params
                lj = j * d + l
                mj = j * d + m
                umjn = u[mj, n]
                du[lj, n] += v * umjn * Bll * Bmm + Alm * umjn
            end
        end

        for i = 1:num_basis_ops
            c_ls_ = c_ls[i]
            c_ms_ = c_ms[i]
            c_vs_ = c_vs[i]
            c = c_func(p, t, i)
            for (m, l, v) in zip(c_ms_, c_ls_, c_vs_)
                Bll = B[l, l]
                Bmm = conj(B[m, m])
                umn = u[m, n]
                du[l, n] += c * v * umn * Bll * Bmm
                for (j, k) in enumerate(opt_param_inds)
                    lj = j * d + l
                    mj = j * d + m
                    du[lj, n] += c * v * u[mj, n] * Bll * Bmm
                    dcdk = ∂c_func(p, t, i, k)
                    du[lj, n] += dcdk * v * umn * Bll * Bmm
                end
            end
        end
    end
    lmul!(-im, du)
end

"""
    make_SE_update_function(sys)

Generate an in-place update function `f!(du,u,p,t)` for the Schrodinger equation based on `sys`.

# Arguments
- `sys::ControllableSystem`: The controllable system 
"""
function make_SE_update_function(sys::ControllableSystem)
    if sys.d_ls === nothing
        if sys.orthogonal_basis
            return (du, u, p, t) -> SE_action!(
                du,
                u,
                p,
                t,
                sys.c_ms,
                sys.c_ls,
                sys.c_vs,
                sys.coefficient_func,
                LinearIndices((1:sys.dim, 1:sys.dim)),
                sys.tol,
            )
        else
            return (du, u, p, t) ->
                SE_action!(du, u, p, t, sys.c_ms, sys.c_ls, sys.c_vs, sys.coefficient_func)
        end
    elseif typeof(sys.reference_frame_generator) <: LinearAlgebra.Diagonal
        return (du, u, p, t) -> SE_action!(
            du,
            u,
            p,
            t,
            sys.d_ms,
            sys.d_ls,
            sys.d_vs,
            sys.c_ms,
            sys.c_ls,
            sys.c_vs,
            sys.coefficient_func,
            sys.reference_frame_generator,
            sys.reference_frame_storage,
        )
    elseif sys.use_rotating_frame == false
        return (du, u, p, t) -> SE_action!(
            du,
            u,
            p,
            t,
            sys.d_ms,
            sys.d_ls,
            sys.d_vs,
            sys.c_ms,
            sys.c_ls,
            sys.c_vs,
            sys.coefficient_func,
        )
    end
end

"""
    make_GOAT_update_function(sys,opt_param_inds)

Generate an in-place update function `f!(du,u,p,t)` for the GOAT equations of motion equation.

# Arguments
- `sys::ControllableSystem`: The controllable system.
- `opt_param_inds::Vector{Int64}`: A vector of indices that specifies which control derivatives will be propogated.
"""
function make_GOAT_update_function(sys::ControllableSystem, opt_param_inds::Vector{Int})
    if sys.d_ls === nothing
        if sys.orthogonal_basis
            return (du, u, p, t) -> GOAT_action!(
                du,
                u,
                p,
                t,
                sys.c_ms,
                sys.c_ls,
                sys.c_vs,
                opt_param_inds,
                sys.coefficient_func,
                sys.∂coefficient_func,
                LinearIndices((1:sys.dim, 1:sys.dim)),
                sys.tol,
            )
        else
            return (du, u, p, t) -> GOAT_action!(
                du,
                u,
                p,
                t,
                sys.c_ms,
                sys.c_ls,
                sys.c_vs,
                opt_param_inds,
                sys.coefficient_func,
                sys.∂coefficient_func,
            )
        end
    elseif typeof(sys.reference_frame_generator) <: LinearAlgebra.Diagonal
        return (du, u, p, t) -> GOAT_action!(
            du,
            u,
            p,
            t,
            sys.d_ms,
            sys.d_ls,
            sys.d_vs,
            sys.c_ms,
            sys.c_ls,
            sys.c_vs,
            opt_param_inds,
            sys.coefficient_func,
            sys.∂coefficient_func,
            sys.reference_frame_generator,
            sys.reference_frame_storage,
        )
    elseif sys.use_rotating_frame == false
        return (du, u, p, t) -> GOAT_action!(
            du,
            u,
            p,
            t,
            sys.d_ms,
            sys.d_ls,
            sys.d_vs,
            sys.c_ms,
            sys.c_ls,
            sys.c_vs,
            opt_param_inds,
            sys.coefficient_func,
            sys.∂coefficient_func,
        )
    end
end

"""
    make_GOAT_initial_state(d,opt_param_inds)

Generate the initial state of the coupled equations of motion for the GOAT method. 
"""
function make_GOAT_initial_state(d, opt_param_inds)
    n_us = size(opt_param_inds, 1) + 1
    u0 = zeros(ComplexF64, d * n_us, d)
    for n = 1:d
        u0[n, n] = 1.0
    end
    return u0
end

"""
    solve_SE(sys, Tmax, p; <keyword arguments>)

Integrate the Schrodinger equation for a specified time and control parameter set. 

# Arguments
- `sys::ControllableSystem`: The controllable system.
- `Tmax::Float64`: The total contorl time.
- `p::Vector{Float64}`: The parameters which define the controlled evolution.
- `ODE_options`: The specification of the integrator settings from OrdinaryDiffEq.jl
"""
function solve_SE(
    sys::ControllableSystem,
    Tmax::Float64,
    p::Vector{Float64};
    t0 = 0.0,
    args...,
)
    tspan = (t0, Tmax)
    f = make_SE_update_function(sys)
    u0 = Matrix{ComplexF64}(I, sys.dim, sys.dim)
    prob = ODEProblem(f, u0, tspan, p)
    sol = solve(prob; args...)
    return sol
end

"""
    solve_GOAT_eoms(sys, opt_param_inds, Tmax, p; <keyword arguments>)

Integrate the Schrodinger equation for a specified time and control parameter set. 

# Arguments
- `sys::ControllableSystem`: The controllable system.
- `opt_param_inds::Vector{Int64}`: The vector of parameter indices specifying which gradients will be propogated.
- `Tmax::Float64`: The total contorl time.
- `p::Vector{Float64}`: The parameters which define the controlled evolution.
- `ODE_options`: The specification of the integrator settings from OrdinaryDiffEq.jl
"""
function solve_GOAT_eoms(
    sys::ControllableSystem,
    opt_param_inds::Vector{Int},
    Tmax::Float64,
    p::Vector{Float64};
    t0 = 0.0,
    args...,
)
    tspan = (t0, Tmax)
    g = make_GOAT_update_function(sys, opt_param_inds)
    u0 = make_GOAT_initial_state(sys.dim, opt_param_inds)
    prob = ODEProblem(g, u0, tspan, p)
    sol = solve(prob; args...)
    return sol
end


"""
    GOAT_infidelity_reduce_map(sys, prob, goat_sol)

Maps the GOAT ODE solution to the objective function and gradient vector using an infidelity measure.

# Arguments
- `sys::ControllableSystem`: The controllable system.
- `prob::QOCProblem`: The QOCProblem
- `goat_sol::OrdinaryDiffEq.solution`: The solution to the GOAT equations of motion.
"""
function GOAT_infidelity_reduce_map(sys::ControllableSystem, prob::QOCProblem, goat_sol)
    d = sys.dim
    Pc = prob.Pc
    Pc_dim = prob.Pc_dim
    target = prob.target
    goatU = goat_sol.u[end]
    n_params = size(goatU, 1) ÷ d
    Ut = goatU[1:d, :]
    g = g_sm(Pc * target * Pc, Ut; dim = Pc_dim)

    ∂g_vec = Float64[]
    for i = 1:n_params-1
        ∂Ut = goatU[i*d+1:(i+1)*d, :]
        ∂g = ∂g_sm(Pc * target * Pc, Ut, ∂Ut; dim = Pc_dim)
        push!(∂g_vec, ∂g)
    end
    return g, ∂g_vec
end

"""
    SE_infidelity_reduce_map(sys, prob, SE_sol)

Maps Schrodinger ODE solution to the objective function using an infidelity measure.

# Arguments
- `sys::ControllableSystem`: The controllable system.
- `prob::QOCProblem`: The QOCProblem
- `SE_sol::OrdinaryDiffEq.solution`: The solution to the Schrodinger equation.
"""
function SE_infidelity_reduce_map(sys::ControllableSystem, prob::QOCProblem, SE_sol)
    d = sys.dim
    Pc = prob.Pc
    Pc_dim = prob.Pc_dim
    target = prob.target
    Ut = SE_sol.u[end]
    g = g_sm(Pc * target * Pc, Ut; dim = Pc_dim)
    return g
end

"""
    solve_GOAT_eoms_reduce(p, sys, prob, opt_param_inds, params::QOCParameters) 

Solves the GOAT eoms and outputs a objective function and gradient vector.

# Arguments
- `p`: The control parameter vector at which the objective and gradient is being calculated. 
- `sys::ControllableSystem`: The controllable system.
- `opt_param_inds`: The vector of parameter indices which determines which gradients are calculated.
- `param::QOCParameters`: The QOCParameters which provides the ODE_options.
"""
function solve_GOAT_eoms_reduce(
    p,
    sys::ControllableSystem,
    prob::QOCProblem,
    opt_param_inds,
    params::QOCParameters,
)
    T = prob.control_time
    goat_sol = solve_GOAT_eoms(sys, opt_param_inds, T, p; params.ODE_options...)
    out = params.GOAT_reduce_map(sys, prob, goat_sol)
    g = first(out)
    ∂gs = last(out)
    return g, ∂gs
end

"""
    parallel_GOAT_fg!(F, G, p, sys, prob, params)

Parallelized computation of the objective and gradient for QOC with GOAT.

# Arguments
- `F`: The objective value
- `G`: The vector of gradients w.r.t. the control parameters `p`. 
- `p`: The control parameter vector.
- `sys::ControllableSystem`: The controllable system.
- `prob::QOCProblem`: The QOCProblem
- `param::QOCParameters`: The QOCParameters.
"""
function parallel_GOAT_fg!(
    F,
    G,
    p,
    sys::ControllableSystem,
    prob::QOCProblem,
    params::QOCParameters,
)
    T = prob.control_time
    if G !== nothing
        num_params = size(p, 1)
        if params.num_params_per_GOAT === nothing
            num_params_per_GOAT = num_params
        else
            num_params_per_GOAT = params.num_params_per_GOAT
        end
        goat_param_indices =
            collect.(collect(Iterators.partition(1:num_params, num_params_per_GOAT)))
        f = y -> solve_GOAT_eoms_reduce(p, sys, prob, y, params)
        out = pmap(f, goat_param_indices)
        gs = first.(out)
        # @assert gs[1] ≈ gs[end] # Trivial sanity check
        for (i, inds) in enumerate(goat_param_indices)
            ∂gs = last(out[i])
            G[inds] .= ∂gs
        end
        g = gs[1]
    else
        sol = solve_SE(sys, T, p; params.ODE_options...)
        g = params.SE_reduce_map(sys, prob, sol)
    end

    if F !== nothing
        return g
    end

end

"""
    parallel_GOAT_fg!(F, G, p, p_storage, opt_param_inds, sys, prob, params)

Parallelized computation of the objective and gradient for QOC with GOAT.

# Arguments
- `F`: The objective value
- `G`: The vector of gradients w.r.t. the control parameters `p`. 
- `p`: The control parameter vector.
- `p_storage`: A pre-allocated storage vector for current `p` values. 
- `opt_param_inds`: The vector of parameter indices which determines which gradients are calculated.
- `sys::ControllableSystem`: The controllable system.
- `prob::QOCProblem`: The QOCProblem
- `param::QOCParameters`: The QOCParameters.
"""
function parallel_GOAT_fg!(
    F,
    G,
    p,
    p_storage,
    opt_param_inds,
    sys::ControllableSystem,
    prob::QOCProblem,
    params::QOCParameters,
)
    T = prob.control_time
    p_storage[opt_param_inds] .= p # Update the storage vector with new parameters from optimization
    if G !== nothing
        num_params = size(p, 1)
        if params.num_params_per_GOAT === nothing
            num_params_per_GOAT = num_params
        else
            num_params_per_GOAT = params.num_params_per_GOAT
        end
        goat_param_indices =
            collect.(collect(Iterators.partition(opt_param_inds, num_params_per_GOAT)))
        f = y -> solve_GOAT_eoms_reduce(p_storage, sys, prob, y, params)
        out = pmap(f, goat_param_indices)
        gs = first.(out)
        # @assert gs[1] ≈ gs[end] # Trivial sanity check
        for i = 1:size(goat_param_indices, 1)
            start = num_params_per_GOAT * (i - 1) + 1
            stop = num_params_per_GOAT * i
            ∂gs = last(out[i])
            G[start:stop] .= ∂gs
        end
        g = gs[1]
    else
        sol = solve_SE(sys, T, p_storage; params.ODE_options...)
        g = params.SE_reduce_map(sys, prob, sol)
    end

    if F !== nothing
        return g
    end

end

"""
    find_optimal_controls(p0, sys, prob, params)

Run the GOAT algorithm and find optimal controls.

# Arguments:
- `p0`: The initial guess of the optimal control parameters.
- `sys::ControllableSystem`: The controllable system.
- `prob::QOCProblem`: The quantum optimal control problem.
- `param::QOCParameters`: The quantum optimal control parameters.
"""
function find_optimal_controls(
    p0,
    sys::ControllableSystem,
    prob::QOCProblem,
    params::QOCParameters,
)
    fg!(F, G, x) = parallel_GOAT_fg!(F, G, x, sys, prob, params)
    res = Optim.optimize(Optim.only_fg!(fg!), p0, params.optim_alg, params.optim_options)
    return res
end

"""
    find_optimal_controls(p0, opt_param_inds, sys, prob, params)

Run the GOAT algorithm and find optimal controls.

# Arguments:
- `p0`: The initial guess of the optimal control parameters.
- `opt_param_inds`: Indices of p0 which specify which parameters to hold constant and which to optimize.
- `sys::ControllableSystem`: The controllable system.
- `prob::QOCProblem`: The quantum optimal control problem.
- `param::QOCParameters`: The quantum optimal control parameters.
"""
function find_optimal_controls(
    p0,
    opt_param_inds,
    sys::ControllableSystem,
    prob::QOCProblem,
    params::QOCParameters,
)
    p_storage = deepcopy(p0)
    fg!(F, G, x) = parallel_GOAT_fg!(F, G, x, p_storage, opt_param_inds, sys, prob, params)
    x0 = p0[opt_param_inds]
    res = Optim.optimize(Optim.only_fg!(fg!), x0, params.optim_alg, params.optim_options)
    return res
end

"""
    evaluate_objective(p, sys, prob, params)

Evaluate the objective function at p.

# Arguments:
- `p`: The optimal control parameters at which to evalute the objective. 
- `sys::ControllableSystem`: The controllable system.
- `prob::QOCProblem`: The quantum optimal control problem.
- `param::QOCParameters`: The quantum optimal control parameters.
"""
function evaluate_objective(
    p::Vector{Float64},
    sys::ControllableSystem,
    prob::QOCProblem,
    params::QOCParameters,
)
    T = prob.control_time
    sol = solve_SE(sys, T, p; params.ODE_options...)
    g = params.SE_reduce_map(sys, prob, sol)
    return g
end

"""
    evaluate_objective(ps, sys, prob, params)

Parallelized evaluation the objective function at multiple control parameters.

# Arguments:
- `ps::Vector{Vector{Float64}}`: The optimal control parameters at which to evalute the objective. 
- `sys::ControllableSystem`: The controllable system.
- `prob::QOCProblem`: The quantum optimal control problem.
- `param::QOCParameters`: The quantum optimal control parameters.
"""
function evaluate_objective(
    ps::Vector{Vector{Float64}},
    sys::ControllableSystem,
    prob::QOCProblem,
    params::QOCParameters,
)
    f = y -> evaluate_objective(y, sys, prob, params)
    out = pmap(f, ps)
    return out
end

end # module
