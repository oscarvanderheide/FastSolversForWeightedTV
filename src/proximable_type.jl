#: Proximable functions

export ProximableFunction, proxy, proxy!, conjugate


# Abstract type

"""Expected behavior for ProximableFunction:
- x = proxy!(y::DT, λ::T, g::ProximableFunction{DT}, x::DT) where {T,N,DT<:AbstractArray{T,N}}
It approximates the solution of the optimization problem: min_x 0.5*norm(x-y)^2+λ*g(x)
"""
abstract type ProximableFunction{T,N} end

function proxy(y::AbstractArray{T,N}, λ::T, g::ProximableFunction{T,N}) where {T,N}
    x = similar(y)
    return proxy!(y, λ, g, x)
end


# Scaled version of proximable functions

struct ScaledProximableFunction{T,N}<:ProximableFunction{T,N}
    c::T
    g::ProximableFunction{T,N}
end

proxy!(y::DT, λ::T, g::ScaledProximableFunction{T,N}, x::DT) where {T,N,DT<:AbstractArray{T,N}} = proxy!(y, λ*g.c, g.g, x)

Base.:*(c::T, g::ProximableFunction{T,N}) where {T,N} = ScaledProximableFunction{T,N}(c, g)
Base.:*(c::T, g::ScaledProximableFunction{T,N}) where {T,N} = ScaledProximableFunction{T,N}(c*g.c, g.g)


# Conjugation of proximable functions

struct ConjugateProximableFunction{T,N}<:ProximableFunction{T,N}
    g::ProximableFunction{T,N}
end

function proxy!(y::Array{T,N}, λ::T, g::ConjugateProximableFunction{T,N}, x::Array{T,N}) where {T,N}
    proxy!(y/λ, T(1)/λ, g.g, x)
    @inbounds for i = 1:length(x)
        x[i] = y[i]-λ*x[i]
    end
    return x
end

function proxy!(y::CuArray{T,N}, λ::T, g::ConjugateProximableFunction{T,N}, x::CuArray{T,N}) where {T,N}
    proxy!(y/λ, T(1)/λ, g.g, x)
    x .= y-λ*x
    return x
end

conjugate(g::ProximableFunction{T,N}) where {T,N} = ConjugateProximableFunction{T,N}(g)
conjugate(g::ConjugateProximableFunction) = g.g