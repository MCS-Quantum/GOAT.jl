using DSP

# """
# Takes in a function for the signal, t0, tf, and the sampling period tp,
# then returns the FT and frequencies in units of 1/[t] (Hz, MHz, GHz, for s, ms, ns). 
# """
# function get_freqs_fft(et,t0,tf,tp)
#     ts = [t0:tp:tf;]
#     ets = et.(ts)
#     F = fft(ets) |> fftshift
#     freqs = fftfreq(length(ts),2*pi/(tp)) |> fftshift 
#     return freqs,F
# end
  

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


"""
Takes in a problem object and returns arrays of timeseries and periodograms for each control channel, K.
"""
function get_controls_and_spectrum(prob)
    K = prob.K
    T = prob.tspan[end]
    ts = [0:0.001:T;]
    series = []
    periodograms = []
    for k in 1:K
        gt = (t) -> prob.ansatz(t,prob,k)
        push!(series, gt.(ts))
        pgam = get_periodogram(gt,ts[1],ts[end],0.001)
        push!(periodograms,pgam)
    end
    return ts, series, periodograms
end