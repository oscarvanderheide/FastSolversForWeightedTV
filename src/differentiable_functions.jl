#: Suite of differentiable functions for TV optimization


# Least-squares misfit

export LeastSquaresMisfit


"""
Least-squares misfit function associated to the problem
min_x 0.5*||A*x-y||^2
"""
struct LeastSquaresMisfit{DT,RT} <: DifferentiableFunction{DT}
    A::AbstractFieldLinearOperator{DT,RT}
    y::RT
end

function grad(f::LeastSquaresMisfit{DT,RT}, x::DT) where {DT,RT}
    T = eltype(x)
    r = f.A*x-f.y
    fx = T(0.5)*norm(r)^2
    gx = adjoint(f.A)*r
    return fx, gx
end