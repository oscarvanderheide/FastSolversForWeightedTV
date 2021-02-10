#: Suite of differentiable functions for TV optimization

export AbstractLeastSquaresMisfit, LeastSquaresMisfit, CuLeastSquaresMisfit, leastsquares_misfit


# Least-squares misfit

"""
Least-squares misfit function associated to the problem
min_x 0.5*||A*x-y||^2
"""
abstract type AbstractLeastSquaresMisfit{T,N1,N2}<:DifferentiableFunction{T,N1} end

struct LeastSquaresMisfit{T,N1,N2}<:AbstractLeastSquaresMisfit{T,N1,N2}
    A::AbstractLinearOperator{Array{T,N1},Array{T,N2}}
    y::Array{T,N2}
end

struct CuLeastSquaresMisfit{T,N1,N2}<:AbstractLeastSquaresMisfit{T,N1,N2}
    A::AbstractLinearOperator{CuArray{T,N1},CuArray{T,N2}}
    y::CuArray{T,N2}
end

function grad!(f::AbstractLeastSquaresMisfit{T,N1,N2}, x::DT, g::Union{Nothing,DT}) where {T,N1,N2,DT<:AbstractArray{T,N1}} 
    r = f.A*x-f.y
    fval = T(0.5)*norm(r)^2
    g !== nothing && (g .= adjoint(f.A)*r)
    return fval
end

leastsquares_misfit(A::AbstractLinearOperator{Array{T,N1},Array{T,N2}}, y::Array{T,N2}) where {T,N1,N2} = LeastSquaresMisfit{T,N1,N2}(A, y)
leastsquares_misfit(A::AbstractLinearOperator{CuArray{T,N1},CuArray{T,N2}}, y::CuArray{T,N2}) where {T,N1,N2} = CuLeastSquaresMisfit{T,N1,N2}(A, y)

(f::AbstractLeastSquaresMisfit{T,N1,N2})(x::DT) where {T,N1,N2,DT<:AbstractArray{T,N1}} = grad!(f, x, nothing)