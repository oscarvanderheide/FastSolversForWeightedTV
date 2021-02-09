#: Differentiable functions

export DifferentiableFunction, grad, grad!


# Abstract type

"""Expected behavior for DifferentiableFunction:
- fval::T = grad!(f::DifferentiableFunction{DT}, x::DT, g::DT) where {T,N,DT<:AbstractArray{T,N}} 
"""
abstract type DifferentiableFunction{DT} end

function grad(f::DifferentiableFunction{DT}, x::DT) where {T,N,DT<:AbstractArray{T,N}}
    g = similar(x)
    fval = grad!(f, x, g)
    return fval, g
end