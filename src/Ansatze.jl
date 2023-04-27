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


"""
    fourier_coefficient(p,t,i,N)

Compute the time dependent coefficient for a control basis operator given by a Fourier ansatze.

# Arguments
- `p`: The vector of control parameters.
- `t`: The time at which to evaluate the coefficient function.
- `i`: The index of the control basis operator.
- `N`: The total number of basis functions used to define the coefficient function. 
"""
function fourier_coefficient(p,t,i,N::Int64)
    c = 0.0
    K = 3
    for n in 1:N
        j = (i-1)*K*N + (n-1)*K # Get the linear index of the ith's control term's nth basis function
        c += p[j+1]*cos(p[j+2]*t+p[j+3])
    end
    return c
end


"""
    ∂fourier_coefficient(p,t,i,l,N)

Computes the partial derivative of `fourier_coefficient` w.r.t p[l] evaluated at (p,t).

# Arguments
- `p`: The vector of control parameters.
- `t`: The time at which to evaluate the coefficient function.
- `i`: The index of the control basis operator.
- `l`: The index of the `p` with which to compute the partial derivative to. 
- `N`: The total number of basis functions used to define the coefficient function. 
"""
function ∂fourier_coefficient(p,t,i,l,N::Int64)
    K = 3
    jmin = (i-1)*K*N+1
    jmax = i*K*N
    if l >= jmin && l <= jmax # Linear indices between the ith control term and the i+1th control terms
        c = 0.0
        for n in 1:N
            j = (i-1)*K*N + (n-1)*K # Get the linear index of the ith's control term's nth basis function
            if j+1 == l
                c += cos(p[j+2]*t+p[j+3])
            elseif j+2 == l
                c += -t*p[j+1]*sin(p[j+2]*t+p[j+3])
            elseif j+3 == l
                c += -p[j+1]*sin(p[j+2]*t+p[j+3])
            end
        end
        return c
    end
    return 0.0
end

"""
    gaussian_coefficient(p,t,i,N)

Compute the time dependent coefficient for a control basis operator given by a gaussian ansatze.

# Arguments
- `p`: The vector of control parameters.
- `t`: The time at which to evaluate the coefficient function.
- `i`: The index of the control basis operator.
- `N`: The total number of basis functions used to define the coefficient function. 
"""
function gaussian_coefficient(p,t,i,N::Int64)
    c = 0.0
    K = 3
    for n in 1:N
        j = (i-1)*K*N + (n-1)*K # Get the linear index of the ith's control term's nth basis function
        c += p[j+1]*exp(-0.5*( (t-p[j+2]) / p[j+3])^2)
    end
    return c
end


"""
    ∂gaussian_coefficient(p,t,i,l,N::Int64)

Computes the partial derivative of `fourier_coefficient` w.r.t p[l] evaluated at (p,t).

# Arguments
- `p`: The vector of control parameters.
- `t`: The time at which to evaluate the coefficient function.
- `i`: The index of the control basis operator.
- `l`: The index of the `p` with which to compute the partial derivative to. 
- `N`: The total number of basis functions used to define the coefficient function. 
"""
function ∂gaussian_coefficient(p,t,i,l,N::Int64)
    K = 3
    jmin = (i-1)*K*N+1
    jmax = i*K*N
    if l >= jmin && l <= jmax # Linear indices between the ith control term and the i+1th control terms
        c = 0.0
        for n in 1:N
            j = (i-1)*K*N + (n-1)*K # Get the linear index of the ith's control term's nth basis function
            if j+1 == l
                c += exp(-0.5*( (t-p[j+2]) / p[j+3])^2)
            elseif j+2 == l
                c += p[j+1]*(t-p[j+2])*exp(-0.5*( (t-p[j+2]) / p[j+3])^2) /(p[j+3]^2)
            elseif j+3 == l
                c += p[j+1]*((t-p[j+2])^2)*exp(-0.5*( (t-p[j+2]) / p[j+3])^2)/(p[j+3]^3)
            end
        end
        return c
    end
    return 0.0
end


"""
    poly_coefficient(p,t,i,N)

Compute the time dependent coefficient for a control basis operator given by a polynomial ansatze.

# Arguments
- `p`: The vector of control parameters.
- `t`: The time at which to evaluate the coefficient function.
- `i`: The index of the control basis operator.
- `N`: The total number of basis functions used to define the coefficient function. 
"""
function poly_coefficient(p,t,i,N::Int64)
    M = 2*N-1 # Number of parameters for each control term
    j0 = (i-1)*M+1
    jstop = i*M
    c = p[j0]
    jstart = j0+1
    for (n,j) in enumerate(jstart:2:jstop)
        c += p[j]*(t-p[j+1])^(n)
    end
    return c
end


"""
    ∂poly_coefficient(p,t,i,l,N)

Computes the partial derivative of `fourier_coefficient` w.r.t p[l] evaluated at (p,t).

# Arguments
- `p`: The vector of control parameters.
- `t`: The time at which to evaluate the coefficient function.
- `i`: The index of the control basis operator.
- `l`: The index of the `p` with which to compute the partial derivative to. 
- `N`: The total number of basis functions used to define the coefficient function. 
"""
function ∂poly_coefficient(p,t,i,l,N::Int64)
    M = 2*N-1 # Number of parameters for each control term
    j0 = (i-1)*M+1
    jstop = i*M
    jrange = j0:jstop
    c = 0.0
    if l==j0
        c += 1.0
    elseif l in jrange
        jstart = j0+1
        for (n,j) in enumerate(jstart:2:jstop)
            if j == l
                c+= (t-p[j+1])^(n)
            elseif j+1 == l
                c+= -p[j]*(n)*(t-p[j+1])^(n)
            end
        end
    end
    return c
end