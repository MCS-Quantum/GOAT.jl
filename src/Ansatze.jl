##### Basic functions
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
    return -( (b-a)/ ( (1+Q*exp(-2*g*x/(b-a)))^2 ))*Q*(exp(-2*g*x/(b-a)))*(-2*g/(b-a))
end

"""
Adds a mixing effect for a local oscillator.
"""
function LO(t,w)
    return cos(w*t)
end



## Ansatze

function constant_ansatz(t,p,k)
    a = p.params
    imap = p.map
    Ns = p.Ns
    @assert length(Ns)==1
    c = a[imap[k,1,1]]
    return c
end

function derivative_constant_ansatz(t,p,k,n,q)
    # If the term in the Hamiltonian is  H_k = c_k*A then ∂/∂c_k H_k = A
    return 1
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

function carrier_fourier_ansatz(t,p,k)
    a = p.params
    imap = p.map
    Ns = p.Ns
    c = 0.0
    # For the channel k the coefficient is defined by
    for n in 1:Ns[k]
        c += (a[imap[k,n,1]]*sin(a[imap[k,n,2]]*t+a[imap[k,n,3]]))*cos(a[imap[k,n,4]]*t)
    end
    return c
end

function derivative_carrier_fourier_ansatz(t,p,k,n,q)
    a = p.params
    imap = p.map
    if q==1
        c = sin(a[imap[k,n,2]]*t+a[imap[k,n,3]])*cos(a[imap[k,n,4]]*t)
        return c
    elseif q==2
        c = a[imap[k,n,1]]*t*cos(a[imap[k,n,2]]*t+a[imap[k,n,3]])*cos(a[imap[k,n,4]]*t)
        return c
    elseif q==3
        c = a[imap[k,n,1]]*cos(a[imap[k,n,2]]*t+a[imap[k,n,3]])*cos(a[imap[k,n,4]]*t)
        return c
    else
        c = -t*sin(a[imap[k,n,4]]*t)*(a[imap[k,n,1]]*sin(a[imap[k,n,2]]*t+a[imap[k,n,3]]))
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


function fourier_coefficient(p,t,i)
    c = 0.0
    for n in 1:N
        j = (i-1)*K*N + (n-1)*K # Get the linear index of the ith's control term's nth basis function
        c += p[j+1]*sin(p[j+2]*t+p[j+3])
    end
    return c
end


"""
Partial deriv of c_func w.r.t the lth element of p
"""
function ∂fourier_coefficient(p,t,i,l)
    jmin = (i-1)*K*N+1
    jmax = i*K*N
    if l >= jmin && l <= jmax # Linear indices between the ith control term and the i+1th control terms
        c = 0.0
        for n in 1:N
            j = (i-1)*K*N + (n-1)*K # Get the linear index of the ith's control term's nth basis function
            if j+1 == l
                c += sin(p[j+2]*t+p[j+3])
            elseif j+2 == l
                c += t*p[j+1]*cos(p[j+2]*t+p[j+3])
            elseif j+3 == l
                c += p[j+1]*cos(p[j+2]*t+p[j+3])
            end
        end
        return c
    end
    return 0.0
end

function gaussian_coefficient(p,t,i)
    c = 0.0
    for n in 1:N
        j = (i-1)*K*N + (n-1)*K # Get the linear index of the ith's control term's nth basis function
        c += p[j+1]*exp(-0.5*( (t-p[j+2]) / p[j+3])^2)
    end
    return c
end


"""
Partial deriv of c_func w.r.t the lth element of p
"""
function gaussian_coefficient(p,t,i,l)
    jmin = (i-1)*K*N+1
    jmax = i*K*N
    if l >= jmin && l <= jmax # Linear indices between the ith control term and the i+1th control terms
        c = 0.0
        for n in 1:N
            j = (i-1)*K*N + (n-1)*K # Get the linear index of the ith's control term's nth basis function
            if j+1 == l
                c += exp(-0.5*( (t-p[j+2]) / p[j+3])^2)
            elseif j+2 == l
                c += p[j+1]*(t-p[j+2]])*exp(-0.5*( (t-p[j+2]) / p[j+3])^2) /(p[j+3]^2)
            elseif j+3 == l
                c += p[j+1]*((t-p[j+2])^2)*exp(-0.5*( (t-p[j+2]) / p[j+3])^2)/(p[j+3]^3)
            end
        end
        return c
    end
    return 0.0
end