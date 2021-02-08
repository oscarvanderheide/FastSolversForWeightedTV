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

function grad!(f::LeastSquaresMisfit{DT,RT}, x::DT, g::DT) where {T,ND,NR,DT<:AbstractArray{T,ND},RT<:AbstractArray{T,NR}}
    r = f.A*x-f.y
    fval = T(0.5)*norm(r)^2
    g .= adjoint(f.A)*r
    return fval
end

leastsquares_misfit(A::AbstractLinearOperator{DT,RT}, y::RT) where {DT,RT} = LeastSquaresMisfit{DT,RT}(A, y)