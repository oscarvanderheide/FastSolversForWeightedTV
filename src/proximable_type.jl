#: Proximable functions

export ProximableFunction, proxy, proxy!, conjugate


# Abstract type

"""Expected behavior for ProximableFunction:
- gval::Union{Nothing,T} = proxy!(y::DT, λ::T, g::ProximableFunction{DT}, x::DT) where {T,N,DT<:AbstractArray{T,N}}
It approximates the solution of the optimization problem: min_x 0.5*norm(x-y)^2+λ*g(x)
"""
abstract type ProximableFunction{DT} end

function proxy(y::DT, λ::T, g::ProximableFunction{DT}) where {T,N,DT<:AbstractArray{T,N}}
    x = similar(y)
    gval = proxy!(y, λ, g, x)
    return gval, x
end


# Scaled version of proximable functions

struct ScaledProximableFunction{DT}<:ProximableFunction{DT}
    c::Number
    g::ProximableFunction{DT}
end

proxy!(y::DT, λ::T, g::ScaledProximableFunction{DT}, x::DT) where {T,N,DT<:AbstractArray{T,N}} = proxy!(y, λ*g.c, g.g, x)

Base.:*(c::T, g::ProximableFunction{DT}) where {T,N,DT<:AbstractArray{T,N}} = ScaledProximableFunction{DT}(c, g)


# Conjugation of proximable functions

struct ConjugateProximableFunction{DT}<:ProximableFunction{DT}
    g::ProximableFunction{DT}
end

function proxy!(y::DT, λ::T, g::ConjugateProximableFunction{DT}, x::DT) where {T,N,DT<:AbstractArray{T,N}}
    gval = proxy!(y/λ, T(1)/λ, g.g, x)
    x .= y-λ*x
    gval === nothing ? (return nothing) : (return -λ^2*gval+T(0.5)*norm(y)^2)
end

conjugate(g::ProximableFunction{DT}) where DT = ConjugateProximableFunction{DT}(g)
conjugate(g::ConjugateProximableFunction) = g.g