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


### Single objective functions
"""
    g_abs(V, U)

Computes the norm of the difference between `V` and `U`.
"""
function g_abs(V, U)
    @assert(size(V) == size(U), "Unitaries are different sizes")
    return 1 - (1 / size(V, 1)) * real(abs(tr(adjoint(V) * U)))
end

"""
    ∂g_abs(V, ∂U)

Computes the partial derivative of `g_abs` w.r.t `U` evaluated at `∂U`
"""
function ∂g_abs(V, ∂U)
    return (-1 / size(V, 1)) * abs(tr(adjoint(V) * ∂U))
end

"""
    g_real(V, ∂U)

Computes infidelity based on the real component of the overlap between `V` and `U`.
"""
function g_real(V, U)
    return 1 - (1 / size(V, 1)) * real(tr(adjoint(V) * U))
end

"""
    ∂g_real(V, ∂U)

Computes the partial derivative of `g_real` w.r.t `U` evaluated at `∂U`.
"""
function ∂g_real(V, ∂U)
    return (1 / size(V, 1)) * real(tr(adjoint(V) * ∂U))
end


"""
    g_sm(V, ∂U)

Computes infidelity via the squared modulus of the overlap between `V` and `U`.

This definition is perhaps the most standard used for quantum optimal control of unitaries.
"""
function g_sm(V, U; dim = 0, args...)
    if dim == 0
        dim = size(V, 1)
    end
    @assert(size(V) == size(U), "Unitaries are different sizes")
    return abs(1 - (1 / dim^2) * real(abs2(tr(adjoint(V) * U))))
end

"""
    ∂g_sm(V, ∂U)

Computes the partial derivative of `g_sm` w.r.t `U` evaluated at `∂U`.
"""
function ∂g_sm(V, U, ∂U; dim = 0, args...)
    if dim == 0
        dim = size(V, 1)
    end
    return (-2 / (dim^2)) * real(tr(adjoint(V) * ∂U) * tr(adjoint(U) * V))
end

"""
    L1(U, P1, P2)

Computes the instantaneous population transferred from between two subspaces.

`P1` and `P2` are projectors onto subspaces 1 and 2.
"""
function L1(U, P1, P2)
    d1 = real(tr(P1))
    return (1 / d1) * real(tr(P2 * U * P1 * adjoint(U)))
end

"""
    h_sm(Us,Pc,Pd,ts)

Computes the total population transferred from between two subspaces during the control time. 
"""
function h_sm(Us, Pc, Pd, ts)
    Tc = ts[end]
    tp = (x) -> L1(x, Pc, Pd)
    ys = tp.(Us)
    return (1 / Tc) * integrate(ts, ys)
end

"""
    ∂h_sm(Us,∂Us,I1,I2,ts)

Computes the partial derivative of `h_sm` w.r.t. `U` evaluated at `∂U` 
"""
function ∂h_sm(Us, ∂Us, I1, I2, ts)
    d1 = real(tr(I1))
    Tc = ts[end]
    tp =
        (x, ∂x) -> real(tr(I1 * adjoint(∂x) * I2 * x * I1 + I1 * adjoint(x) * I2 * ∂x * I1))
    ys = tp.(Us, ∂Us)
    return (1 / Tc) * (1 / d1) * integrate(ts, ys)
end
