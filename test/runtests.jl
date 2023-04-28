using GOAT


ts = [0:0.00001:100;];
tau = 1.2352
N = 3
p0 = rand(N * 3)
s(t) = fourier_coefficient(p0, t, 1, N)
signal = s.(ts)
as, fs, phis = get_sinusoidal_basis_parameters(ts, signal)
s_fft(t) = time_domain_signal(t, as, fs, phis)
@assert isapprox(s(tau), s_fft(tau); atol = 1e-5)
p = [y for x in zip(as, fs, phis) for y in x]
s_ansatz(t) = fourier_coefficient(p, t, 1, length(as))
@assert isapprox(s(tau), s_ansatz(tau); atol = 1e-5)
