using Cuba, NumericalIntegration



### Single objective functions
function g_abs(U_target, U)
    @assert(size(U_target)==size(U),"Unitaries are different sizes")
    return 1-(1/size(U_target,1))*real(abs(tr(adjoint(U_target)*U)))
end

function ∂g_abs(U_target, ∂U)
    return (-1/size(U_target,1))*abs(tr(adjoint(U_target)*∂U))
end

function g_real(U_target,U)
    return 1-(1/size(U_target,1))*real(tr(adjoint(U_target)*U))
end

function ∂g_real(U_target,∂U)
    return (1/size(U_target,1))*real(tr(adjoint(U_target)*∂U))
end

function g_sm(U_target, U; dim=0, args...)
    if dim==0
        dim = size(U_target,1)
    end
    @assert(size(U_target)==size(U),"Unitaries are different sizes")
    return abs(1-(1/dim^2)*real(abs2(tr(adjoint(U_target)*U))))
end

function ∂g_sm(U_target, U, ∂U; dim=0, args...)
    if dim==0
        dim = size(U_target,1)
    end
    return (-2/(dim^2))*real(tr(adjoint(U_target)*∂U)*tr(adjoint(U)*U_target))
end

function L1(U, I1, I2)
    d1 = real(tr(I1))
    return (1/d1)*real(tr(I2*U*I1*adjoint(U)))
end

function h_sm(Us,Pc,Pd,ts)
    Tc = ts[end]
    tp = (x) -> L1(x,Pc,Pd)
    ys = tp.(Us)
    return (1/Tc)*integrate(ts,ys)
end

function ∂h_sm(Us,∂Us,I1,I2,ts)
    d1 = real(tr(I1))
    Tc = ts[end]
    tp = (x,∂x) -> real(tr( I1*adjoint(∂x)*I2*x*I1 + I1*adjoint(x)*I2*∂x*I1))
    ys = tp.(Us,∂Us)
    return (1/Tc)*(1/d1)*integrate(ts,ys)
end



## Flexible/multiple objective functions
function geometric_mean(xs)
    m = 0
    for x in xs
        m+=log(x)
    end
    return exp(m/(length(xs)))
end


function J1(Us::Vector,V,fs::Vector, g::Function; args...)
    J = 0
    for U in Us
        for f in fs
            J+=log(g(U,f(V); args...))
        end
    end
    return exp(J/(length(Us)*length(fs)))
end


function J1(U,V, fs::Vector, g::Function; args...)
    J = 0
    for f in fs
        J+=log(g(U,f(V); args...))
    end
    return exp(J/length(fs))
end



function J2(U::Matrix,V::Matrix,f::Function, theta_lims::Array, g::Function; atol=1e-2,rtol=1e-8, args...)
    dim_theta = length(theta_lims)
    lowers = first.(theta_lims)
    uppers = last.(theta_lims)
    function integrand_average(y,z) # Integrand is already scaled by the volume of the domain so integrating yields an average
        thetas = (lowers.+(uppers.-lowers).*y)
        FV = f(V,thetas)
        z[1] = log(g(U,FV;args...))
    end
    integral, error, probability, neval, fail, nregions = divonne(integrand_average,dim_theta,1;rtol=rtol,atol=atol)
    return exp(integral[1])
end

function J2(Us::Vector,V::Matrix,f::Function, theta_lims::Array, g::Function; atol=1e-3, rtol=1e-8, args...)
    exponent = 0
    dim_theta = length(theta_lims)
    lowers = first.(theta_lims)
    uppers = last.(theta_lims)
    function integrand_average(U,y,z) # Integrand is already scaled by the volume of the domain so integrating yields an average
        thetas = (lowers.+(uppers.-lowers).*y)
        FV = f(V,thetas)
        z[1] = log(g(U,FV;args...))
    end
    for U in Us
        integral, error, probability, neval, fail, nregions = divonne((y,z)-> integrand_average(U,y,z),dim_theta,1;rtol=rtol,atol=atol)
        exponent+= integral[1]
    end
    
    return exp(exponent/length(Us))
end

function ∂g_smUnfmV(U::Matrix, V::Matrix, ∂V::Matrix, f::Function, ∂f::Function; dim=0, args...)
    fmV = f(V)
    ∂fmV = ∂f(V,∂V)
    if dim==0
        dim = size(U,1)
    end
    return (-2/(dim^2))*real(tr(adjoint(U)*∂fmV)*tr(adjoint(fmV)*U))
end

function ∂g_smUnfθV(U::Matrix, V::Matrix, ∂V::Matrix, f::Function, ∂f::Function, thetas::Vector; dim=0, args...)
    fmV = f(V,thetas)
    ∂fmV = ∂f(V,∂V,thetas)
    if dim==0
        dim = size(U,1)
    end
    return (-2/(dim^2))*real(tr(adjoint(U)*∂fmV)*tr(adjoint(fmV)*U))
end

function ∂J1(U::Matrix, V::Matrix, ∂V::Matrix, fs::Vector, ∂fs::Vector, g::Function, ∂gUnfmV::Function ; args...)
    J1_val = J1(U,V,fs,g ; args...)
    M = length(fs)
    c = 0
    for (f,∂f) in zip(fs,∂fs)
        c+= ∂gUnfmV(U,V,∂V,f, ∂f; args...)/(g(U,f(V)))
    end
    return J1_val*c/M
end

function ∂J1(Us::Vector, V::Matrix, ∂V::Matrix, fs::Vector, ∂fs::Vector, g::Function, ∂gUnfmV::Function ; args...)
    J1_val = J1(U,V,fs,g ; args...)
    M = length(fs)
    c = 0
    for U in Us
        for (f,∂f) in zip(fs,∂fs)
            c+= ∂gUnfmV(U,V,∂V,f, ∂f; args...)/ g(U,f(V); args...)
        end
    end
    return J1_val*c/(M*length(Us))
end

function ∂J2(U::Matrix, V::Matrix, ∂V::Matrix, f::Function, ∂f::Function, theta_lims::Array, g::Function, ∂gUnfθV::Function ; atol=1e-3,rtol=1e-8, maxevals=10000, args...)
    J2_val = J2(U,V,f,theta_lims,g ; rtol=rtol, atol=atol, args...)
    dim_theta = length(theta_lims)
    lowers = first.(theta_lims)
    uppers = last.(theta_lims)
    
    function integrand_average(y,z) # Integrand is already scaled by the volume of the domain so integrating yields an average
        thetas = (lowers.+(uppers.-lowers).*y)
        fV = f(V,thetas)
        z[1] = ∂gUnfθV(U,V,∂V,f,∂f,thetas; args...) / g(U,fV; args...)
    end
    
    integral, error, probability, neval, fail, nregions = divonne(integrand_average,dim_theta,1;rtol=rtol,atol=atol, maxevals=maxevals)
    c = integral[1]
    return J2_val*c
end
