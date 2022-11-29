##### Basic functions
function sinusoid_kernel(t, a, w, phi)
    return a*sin(w*t+phi)
end

function gaussian_kernel(t, a, mu,sigma)
    return a*exp(-0.5*( (t-mu) / sigma)^2)
end


"""
Morlet wavelet kernel (although not a true wavelet because it is not normalized):

morlet_kernel(t, a, mu, sigma, w, phi)
"""
function morlet_kernel(t, a, mu, sigma, w, phi)
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
    flat_top_cosine(t, A, T, tr)

A flat-top cosine function with duration T
and a rise/fall time of tr with a cosine-shaped rise/fall.

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
Partial deriv of c_func w.r.t the lth element of p
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
Partial deriv of c_func w.r.t the lth element of p
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
Partial deriv of c_func w.r.t the lth element of p
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