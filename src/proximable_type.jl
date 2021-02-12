#: Proximable functions

export ProximableFunction, proxy, proxy!, project, project!, conjugate


# Abstract type

"""Expected behavior for ProximableFunction:
- proxy!(y::DT, λ::T, g::ProximableFunction{DT}, x::DT) where {T,N,DT<:AbstractArray{T,N}}
It approximates the solution of the optimization problem: min_x 0.5*norm(x-y)^2+λ*g(x)
- project!(y::DT, ε::T, g::ProximableFunction{DT}, x::DT) where {T,N,DT<:AbstractArray{T,N}}
It project min_{g(x)<=ε}0.5*||x-y||^2
"""
abstract type ProximableFunction{T,N} end

proxy(y::AbstractArray{T,N}, λ::T, g::ProximableFunction{T,N}) where {T,N} = proxy!(y, λ, g, similar(y))
project(y::AbstractArray{T,N}, ε::T, g::ProximableFunction{T,N}) where {T,N} = project!(y, ε, g, similar(y))


# Scaled version of proximable functions

struct ScaledProximableFunction{T,N}<:ProximableFunction{T,N}
    c::T
    g::ProximableFunction{T,N}
end

proxy!(y::DT, λ::T, g::ScaledProximableFunction{T,N}, x::DT) where {T,N,DT<:AbstractArray{T,N}} = proxy!(y, λ*g.c, g.g, x)
project!(y::DT, ε::T, g::ScaledProximableFunction{T,N}, x::DT) where {T,N,DT<:AbstractArray{T,N}} = proxy!(y, ε/g.c, g.g, x)


# LinearAlgebra

Base.:*(c::T, g::ProximableFunction{T,N}) where {T,N} = ScaledProximableFunction{T,N}(c, g)
Base.:*(c::T, g::ScaledProximableFunction{T,N}) where {T,N} = ScaledProximableFunction{T,N}(c*g.c, g.g)


# Conjugation of proximable functions

struct ConjugateProximableFunction{T,N}<:ProximableFunction{T,N}
    g::ProximableFunction{T,N}
end

function proxy!(y::DT, λ::T, g::ConjugateProximableFunction{T,N}, x::DT) where {T,N,DT<:AbstractArray{T,N}}
    proxy!(y/λ, T(1)/λ, g.g, x)
    x .= y-λ*x
    return x
end

conjugate(g::ProximableFunction{T,N}) where {T,N} = ConjugateProximableFunction{T,N}(g)
conjugate(g::ConjugateProximableFunction) = g.g