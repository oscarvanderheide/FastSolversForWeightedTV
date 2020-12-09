#: Solver types

export DifferentiableFunction, ProximableFunction
export grad, grad!, proxy, proxy!, conjugate


# Abstract types

# We assume a fval,g=grad(f,x) is implemented
abstract type DifferentiableFunction{DT} end # f : DT -> R
function grad!(g_::DT, f::DifferentiableFunction{DT}, x::DT) where DT
    fval, g = grad(f, x)
    update!(g_, g)
    return fval
end

# We assume a gval,y=proxy(λ,g,x), where gval=g(x) is implemented
abstract type ProximableFunction{DT} end # g : DT -> R, always assumed convex!
function proxy!(y_::DT, λ::T, g::ProximableFunction{DT}, x::DT) where {T,DT}
    gval, y = proxy(λ, g, x)
    update!(y_, y)
    return gval
end


# Scaled version of proximable functions

struct ScaledProximableFunction{DT} <: ProximableFunction{DT}
    c::Number
    g::ProximableFunction{DT}
end
proxy(λ::T, g::ScaledProximableFunction{DT}, x::DT) where {T,DT<:AbstractField2D{T}} = proxy(λ*g.c, g.g, x)

*(c::Number, g::ProximableFunction{DT}) where DT = ScaledProximableFunction{DT}(c, g)


# Conjugation of proximable functions

struct ConjugateProximableFunction{DT} <: ProximableFunction{DT} # g^* : DT -> R
    g::ProximableFunction{DT}
end
function proxy(λ::T, g::ConjugateProximableFunction{DT}, y::DT) where {T,DT<:AbstractField2D{T}}
    return nothing, y-λ*proxy(T(1)/λ, g.g, y/λ)[2]
end

conjugate(g::ProximableFunction{DT}) where DT = ConjugateProximableFunction{DT}(g)
conjugate(g::ConjugateProximableFunction{DT}) where DT = g.g