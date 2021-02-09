#: Suite of differentiable functions for TV optimization


# Least-squares misfit

export LeastSquaresMisfit, leastsquares_misfit


"""
Least-squares misfit function associated to the problem
min_x 0.5*||A*x-y||^2
"""
struct LeastSquaresMisfit{DT,RT}<:DifferentiableFunction{DT}
    A::AbstractLinearOperator{DT,RT}
    y::RT
end

function grad!(f::LeastSquaresMisfit{DT,RT}, x::DT, g::DT) where {T,N,DT<:AbstractArray{T,N},RT} 
    r = f.A*x-f.y
    fval = T(0.5)*norm(r)^2
    g .= adjoint(f.A)*r
    return fval
end

leastsquares_misfit(A::AbstractLinearOperator{DT,RT}, y::RT) where {T,N1,N2,DT<:AbstractArray{T,N1},RT<:AbstractArray{T,N2}} = LeastSquaresMisfit{DT,RT}(A, y)