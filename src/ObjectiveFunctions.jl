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