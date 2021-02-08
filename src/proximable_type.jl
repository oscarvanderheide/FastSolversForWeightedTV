#: Proximable functions

export ProximableFunction, proxy, proxy!, conjugate


# Abstract type

"""Expected behavior for ProximableFunction:
- gval::Union{Nothing,T} = proxy!(y::AbstractArray{T,N}, λ::T, g::ProximableFunction{T,N}, x::AbstractArray{T,N})
It approximates the solution of the optimization problem: min_x 0.5*norm(x-y)^2+λ*g(x)
"""
abstract type ProximableFunction{T,N} end

function proxy(y::AbstractArray{T,N}, λ::T, g::ProximableFunction{T,N}) where {T,N}
    x = similar(y)
    gval = proxy!(y, λ, g, x)
    return gval, x
end


# Scaled version of proximable functions

struct ScaledProximableFunction{T,N}<:ProximableFunction{T,N}
    c::T
    g::ProximableFunction{T,N}
end

proxy!(y::AbstractArray{T,N}, λ::T, g::ScaledProximableFunction{T,N}, x::AbstractArray{T,N}) where {T,N} = proxy!(y, λ*g.c, g.g, x)

*(c::T, g::ProximableFunction{T,N}) where {T,N} = ScaledProximableFunction{T,N}(c, g)


# Conjugation of proximable functions

struct ConjugateProximableFunction{T,N}<:ProximableFunction{T,N}
    g::ProximableFunction{T,N}
end

function proxy!(y::AbstractArray{T,N}, λ::T, g::ConjugateProximableFunction{T,N}, x::AbstractArray{T,N}) where {T,N}
    gval = proxy!(y/λ, T(1)/λ, g.g, x)
    x .= y-λ*x
    gval === nothing ? (return nothing) : (return -λ^2*gval+T(0.5)*norm(y)^2)
end

conjugate(g::ProximableFunction{T,N}) where {T,N} = ConjugateProximableFunction{T,N}(g)
conjugate(g::ConjugateProximableFunction{T,N}) where {T,N} = g.g