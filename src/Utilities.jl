using FFTW, Random, Printf


"""
    get_sinusoidal_coefficients_from_FFT(ts, s)

Compute the sinusoidal amplitude-phase coefficients and frequencies from FFT of the signal s with timeseries ts. 
"""
function get_sinusoidal_coefficients_from_FFT(ts, s)
    P = ts[end]
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
    truncated_inv_fft(t, Aks, phi_ks, freqs ; N=nothing)

Compute the signal at time t defined by the sinusoidal amplitude-phase coefficients and FFT frequencies truncated
to the N frequency components with largest magnitude.
"""
function truncated_inv_fft(t, Aks, phi_ks, freqs ; N=nothing)
    c = 0.0
    if N === nothing
        for (Ak, phi_k, f) in zip(Aks, phi_ks, freqs)
            c += Ak*cos(f*t+phi_k)
        end
    else
        I = sortperm(abs.(Aks), rev=true)[1:N]
        for i in I
            c+= Aks[i]*cos(freqs[i]*t+phi_ks[i])
        end
    end
    return c
end

function time_domain_signal(t::Float64, amps, freqs, phases)
    c = 0.0
    for (a,f,p) in zip(amps, freqs, phases)
        c += sin(t*2*pi*f+2*pi*p)*a
    end
    return c
end


function colored_noise(lf::Float64, hf::Float64, n::Int64, alpha::Float64, seed::Int64)
    freqs = collect(range(lf,hf,length=n))
    amps = freqs.^alpha
    phases = rand(MersenneTwister(seed), Float64, size(freqs,1))
    return amps, freqs, phases
end 

function test_derivatives(sys, prob, opt_param_inds, p_test; dh=1e-8, tol=1e-5, diffeq_options = (abstol = 1e-9, reltol= 1e-9, alg=Vern9()), SE_reduce_map = SE_infidelity_reduce_map, GOAT_reduce_map=GOAT_infidelity_reduce_map, only_coefficeint_funcs=true)
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
            g = evaluate_infidelity(p_test,sys,prob,SE_reduce_map,diffeq_options)
            dg = evaluate_infidelity(p,sys,prob,SE_reduce_map,diffeq_options)
            ∂g_FD = (dg-g)/dh
            g,∂g = solve_GOAT_eoms_reduce(p_test,sys,prob,[j], GOAT_reduce_map, diffeq_options)
            diff = abs(∂g_FD - ∂g[1])
            strdiff = @sprintf "%.5e" diff
            @assert diff <= tol "Error in the unitary gradient of basis $i parameter $j: $strdiff"
            #println("Gradient acceptable for parameter $j of basis $i")
        end
    end
    return println("All good")
end