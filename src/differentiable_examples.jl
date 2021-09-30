#: Suite of differentiable functions for TV optimization

export LeastSquaresMisfit, leastsquares_misfit


# Least-squares misfit

"""
Least-squares misfit function associated to the problem
min_x 0.5*||A*x-y||^2
"""
struct LeastSquaresMisfit{T,N,AT<:AbstractLinearOperator,YT<:AbstractArray}<:DifferentiableFunction{T,N}
    A::AT
    y::YT
end

leastsquares_misfit(A::AbstractLinearOperator{DT,RT}, y::RT) where {T,N1,N2,DT<:AbstractArray{T,N1},RT<:AbstractArray{T,N2}} = LeastSquaresMisfit{T,N1,typeof(A),RT}(A, y)

function grad!(f::LeastSquaresMisfit{T,N,AT,YT}, x::AbstractArray{T,N}, g::Union{Nothing,AbstractArray{T,N}}) where {T,N,AT,YT}
    r = f.A*x-f.y
    fval = T(0.5)*norm(r)^2
    g !== nothing && (g .= adjoint(f.A)*r)
    return fval
end