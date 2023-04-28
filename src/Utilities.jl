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


using FFTW, Random, Printf


"""
    get_sinusoidal_basis_parameters(ts, s)

Compute the sinusoidal amplitude, frequencies, and phases of a signal.

Outputs a triplet of: Amps, freqs, phases via the decomposition: ``s(t) = ∑ᵢ aᵢ sin(ωᵢt+ϕᵢ)``.

# Arguments
- `ts`: the sampling times
- `s`: the signal at each sampling time
"""
function get_sinusoidal_basis_parameters(ts, s)
    dt = ts[2]-ts[1]
    fs = 1/dt
    N = size(ts,1)
    Fs = fft(s)[1:N÷2]
    freqs = 2*pi*fftfreq(length(s),fs)
    ak = (2/N)*real.(Fs)
    bk = (-2/N)*imag.(Fs)
    ak[1] = ak[1]/2
    phi_k = atan.(bk./ak)
    Ak = ak ./ cos.(phi_k)
    return Ak, freqs[1:size(Ak,1)], -phi_k
end


"""
    time_domain_signal(t::Float64, amps, freqs, phases, N)

Compute a truncatsed inverse fourier transform at time `t`.

The signal ``s(t)`` is reconstructed via the function ``s(t) = ∑ᵢ aᵢ cos(ωᵢt+ϕᵢ)``

# Arguments
- `t`: The time to evaluate the inverse FFT.
- `amps`: The amplitudes.
- `freqs`: The frequencies.
- `phases`: The phases.
- `N`: The number of components to keep when reconstructing the signal.
"""
function time_domain_signal(t::Float64, amps, freqs, phases, N)
    c = 0.0
    I = sortperm(abs.(amps), rev=true)[1:N]
    for i in I
        c+= amps[i]*cos(freqs[i]*t+phases[i])
    end
    return c
end

"""
    time_domain_signal(t::Float64, amps, freqs, phases) 

Compute a truncated inverse fourier transform at time `t`.

The signal ``s(t)`` is reconstructed via the function ``s(t) = ∑ᵢ aᵢ sin(ωᵢt+ϕᵢ)``

# Arguments
- `t`: The time to evaluate the inverse FFT.
- `Aks`: The amplitudes.
- 'freqs`: The frequencies.
- `phi_ks`: The phases.
"""
function time_domain_signal(t::Float64, amps, freqs, phases)
    c = 0.0
    for (a,f,p) in zip(amps, freqs, phases)
        c += a*cos(t*f+p)
    end
    return c
end


"""
    colored_noise(lf, hf, n, alpha, seed)

Generate a set of amplitude, frequencies, and phases for randomly generated colored noise. 

Specifically, generates colored noise with P(ω) ∝ ωᵅ

# Arguments
- `lf`: The low frequency cutoff. 
- `hf`: The high frequency cutoff.
- 'n`: The number of frequncy components.
- `alpha`: The color of the noise.
"""
function colored_noise(lf::Float64, hf::Float64, n::Int64, alpha::Float64, seed::Int64)
    freqs = collect(range(lf,hf,length=n))
    amps = freqs.^alpha
    phases = rand(MersenneTwister(seed), Float64, size(freqs,1))
    return amps, freqs, phases
end 



"""
    test_derivatives(sys, prob, params, opt_param_inds, p_test; <keyword arguments>)

Uses a finite difference method to confirm that gradients are calculated correctly.

If `only_coefficeint_funcs=true` then only the coefficient functions are checked.
If `only_coefficeint_funcs=false` then the GOAT equations of motion 
are solved and unitary gradients are checked too. 

# Arguments
- `sys`: The `ControllableSystem`.
- `prob`: The `QOCProblem`.
- 'params`: The `QOCParameters`
- `dh=1e-8`: The finite-difference step size of each parameter.
- `tol=1e-5`: The tolerance that determines whether an error is raised. 
- `only_coefficeint_funcs=true`: Specifies if checking gradients of coefficients or unitaries. 
"""
function test_derivatives(sys, prob, params, opt_param_inds, p_test; dh=1e-8, tol=1e-5, only_coefficeint_funcs=true)
    num_basis_funcs = size(sys.c_ls,1)
    p = similar(p_test)
    for j in opt_param_inds
        for i in 1:num_basis_funcs
            p .= p_test
            c = sys.coefficient_func(p,1.0,i)
            ∂c = sys.∂coefficient_func(p,1.0,i,j)
            p[j] += dh
            dc = sys.coefficient_func(p,1.0,i)
            ∂c_FD = (dc-c)/dh
            diff = abs(∂c_FD - ∂c)
            strdiff = @sprintf "%.5e" diff
            @assert diff <= tol "Error in the coefficient gradient of basis $i parameter $j: $strdiff"
            #println("Function gradient acceptable for parameter $j of basis $i")
            if only_coefficeint_funcs
                continue
            end
            g = evaluate_infidelity(p_test,sys,prob,params)
            dg = evaluate_infidelity(p,sys,prob,params)
            ∂g_FD = (dg-g)/dh
            g,∂g = solve_GOAT_eoms_reduce(p_test,sys,prob,[j], params)
            diff = abs(∂g_FD - ∂g[1])
            strdiff = @sprintf "%.5e" diff
            @assert diff <= tol "Error in the unitary gradient of basis $i parameter $j: $strdiff"
            #println("Gradient acceptable for parameter $j of basis $i")
        end
    end
    return println("Derivatives are all good")
end

##### Basic functions

"""
    sinusoid_kernel(t, a, w, phi)

Computes the value of a sine function. 
"""
function sinusoid_kernel(t, a, w, phi)
    return a*sin(w*t+phi)
end

"""
    gaussian_kernel(t, a, mu,sigma)

Computes the value of a gaussian. 
"""
function gaussian_kernel(t, a, mu,sigma)
    return a*exp(-0.5*( (t-mu) / sigma)^2)
end


"""
    morlet_kernel(t, a, mu, sigma, w, phi)

Morlet wavelet kernel (although not a true wavelet because it is not normalized).

Effectively a gaussian-windowed sinusoid function. 
"""
function morlet_kernel(t, a, mu, sigma, w, phi)
    return a*exp(-0.5*( (t-mu) / sigma)^2)*sin(w*t+phi)
end

"""
    general_logistic(t, lower, upper, slope, start, nu)

A version of a generalized logistic function found on wikipedia: 
https://en.wikipedia.org/wiki/Generalised_logistic_function.

# Arguments
- `t`: The time.
- `lower`: The lower asymptote.
- `upper`: The upper asymptote.
- `slope` : The slope of the function.
- `start = 1e3`: A shift of the logistic function to adjust `general_logistic(t=0)`.
- `nu = 1.0`: A parameter to set where the maximum derivative is located. 
"""
function general_logistic(t, lower, upper, slope, start=1e3, nu=1.0)
    A = lower
    B = slope
    K = upper
    Q = start # The larger this is the smaller general_logistic(t=0)
    return A + (K-A)/((1+Q*exp(-B*t))^(1/nu))
end

"""
    flat_top_cosine(t, A, T, tr)

A flat-top cosine function kernel. 

# Arguments
- `A`: The amplitude of the function.
- `T`: Duration of the function.
- `tr`: The rise and fall time.
"""
function flat_top_cosine(t, A, T, tr)
    if t<=tr
        e = 0.5*A*(1-cos(pi*t/tr))
        return e
    elseif t>=(T-tr)
        e = 0.5*A*(1-cos(pi*(T-t)/tr))
        return e
    else
        e = A
        return e
    end
end


"""
    window(x,lower,upper,gradient)

A windowing function based on a product of two generalized logistic functions.

See `general_logistic` for additional details.

# Arguments
- `lower`: The location of the left tail.
- `right`: The location of the right tail.
- `gradient`: The maximum gradient of the logistic functions
"""
function window(x,lower,upper,gradient=20)
    rising = general_logistic(x-lower,0,1,gradient)
    lowering = general_logistic(-x+upper,0,1,gradient)
    return lowering*rising
end


"""
    S(x, lower, upper ; gradient=1)

A saturation function that limits amplitudes to a particular range specified by [lower,upper].
"""
function S(x, lower,upper ; gradient=1)
    mid = (upper-lower)*0.5
    Q = -upper/lower
    return general_logistic(x/mid,lower,upper,gradient,Q)
end


"""
    dSdx(x, lower, upper; gradient=1)

Calculates the partial derivative of the saturation function w.r.t the independent variable x.
"""
function dSdx(x, lower, upper ; gradient=1)
    mid = (upper-lower)*0.5
    Q = -upper/lower
    b = upper
    a = lower
    g = gradient
    xbar = x/mid
    return -( (b-a)/ ( (1+Q*exp(-2*g*x/(b-a)))^2 ))*Q*(exp(-2*g*x/(b-a)))*(-2*g/(b-a))
end

"""
    LO(t,w)

Simulates a local oscillator with frequency `w`.
"""
function LO(t,w)
    return cos(w*t)
end