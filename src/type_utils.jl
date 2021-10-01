#: Utils

export conjugate, proxy_objfun, proj_objfun
export no_constraints, indicator


# Scaled version of proximable/projectionable functions

struct ScaledProximableFun{T,N}<:ProximableFunction{T,N}
    c::T
    g::ProximableFunction{T,N}
end

proxy!(y::AbstractArray{T,N}, λ::T, g::ScaledProximableFun{T,N}, x::AbstractArray{T,N}) where {T,N} = proxy!(y, λ*g.c, g.g, x)
project!(y::AbstractArray{T,N}, ε::T, g::ScaledProximableFun{T,N}, x::AbstractArray{T,N}) where {T,N} = project!(y, g.c/ε, g.g, x)


# LinearAlgebra

Base.:*(c::T, g::ProximableFunction{T,N}) where {T,N} = ScaledProximableFun{T,N}(c, g)
Base.:*(c::T, g::ScaledProximableFun{T,N}) where {T,N} = ScaledProximableFun{T,N}(c*g.c, g.g)
Base.:/(g::ProximableFunction{T,N}, c::T) where {T,N} = ScaledProximableFun{T,N}(T(1)/c, g)
Base.:/(g::ScaledProximableFun{T,N}, c::T) where {T,N} = ScaledProximableFun{T,N}(g.c/c, g.g)


# Conjugation of proximable functions

struct ConjugateProxFun{T,N}<:ProximableFunction{T,N}
    g::ProximableFunction{T,N}
end

function proxy!(y::AbstractArray{T,N}, λ::T, g::ConjugateProxFun{T,N}, x::AbstractArray{T,N}) where {T,N}
    proxy!(y/λ, T(1)/λ, g.g, x)
    x .= y-λ*x
    return x
end

conjugate(g::ProximableFunction{T,N}) where {T,N} = ConjugateProxFun{T,N}(g)
conjugate(g::ConjugateProxFun) = g.g


# Constraint sets

## No constraints

struct NoConstraints{T,N}<:ProjectionableSet{T,N} end

no_constraints(T::DataType, N::Int64) = NoConstraints{T,N}()

project!(x::AbstractArray{T,N}, ::NoConstraints{T,N}, y::Array{T,N}) where {T,N} = (y .= x)

## Sub-level sets with proximable functions

"""
Constraint set C = {x:g(x)<=ε}
"""
struct SubLevelSet{T,N}<:ProjectionableSet{T,N}
    g::ProximableFunction{T,N}
    ε::T
end

Base.:≤(g::ProximableFunction{T,N}, ε::T) where {T,N} = SubLevelSet{T,N}(g, ε)

project!(x::AbstractArray{T,N}, C::SubLevelSet{T,N}, y::AbstractArray{T,N}) where {T,N} = project!(x, C.ε, C.g, y)
project!(x::AbstractArray{T,N}, C::SubLevelSet{T,N}, y::AbstractArray{T,N}, opt::OptimOptions) where {T,N} = project!(x, C.ε, C.g, y, opt)

## Indicator function

"""
Indicator function δ_C(x) = {0, if x ∈ C; ∞, otherwise} for convex sets C
"""
struct IndicatorFunction{T,N}<:ProximableFunction{T,N}
    C::ProjectionableSet{T,N}
end

indicator(C::ProjectionableSet{T,N}) where {T,N} = IndicatorFunction{T,N}(C)

proxy!(y::AbstractArray{T,N}, ::T, δ::IndicatorFunction{T,N}, x::AbstractArray{T,N}) where {T,N} = project!(y, δ.C, x)
proxy!(y::AbstractArray{T,N}, ::T, δ::IndicatorFunction{T,N}, x::AbstractArray{T,N}, opt::OptimOptions) where {T,N} = project!(y, δ.C, x, opt)



# Proximable + linear operator

struct WeightedProximableFun{T,N1,N2}<:ProximableFunction{T,N1}
    g::ProximableFunction{T,N2}
    A::AbstractLinearOperator{<:AbstractArray{T,N1},<:AbstractArray{T,N2}}
end

Base.:∘(g::ProximableFunction{T,N2}, A::AbstractLinearOperator{DT,RT}) where {T,N1,N2,DT<:AbstractArray{T,N1},RT<:AbstractArray{T,N2}} = WeightedProximableFun{T,N1,N2}(g, A)

function proxy!(y::AbstractArray{T,N1}, λ::T, g::WeightedProximableFun{T,N1,N2}, x::AbstractArray{T,N1}, opt::OptimOptions) where {T,N1,N2}

    # Objective function (dual problem)
    gT = conjugate(g.g)/λ
    f = leastsquares_misfit(adjoint(g.A), y/λ)+gT

    # Minimization (dual variable)
    p0 = proxy(opt.steplength/λ*g.A*y, T(1)/λ, gT)
    # p0 = T(0)*proxy(opt.steplength/λ*g.A*y, T(1)/λ, gT)
    p = minimize(f, p0, opt)

    # Dual to primal solution
    return x .= y-λ*adjoint(g.A)*p

end

function project!(y::AbstractArray{T,N1}, ε::T, g::WeightedProximableFun{T,N1,N2}, x::AbstractArray{T,N1}, opt::OptimOptions) where {T,N1,N2}

    # Objective function (dual problem)
    gT = conjugate(indicator(g.g ≤ ε))
    f = leastsquares_misfit(adjoint(g.A), y)+gT

    # Minimization (dual variable)
    p0 = proxy(opt.steplength*g.A*y, T(1), gT)
    p = minimize(f, p0, opt)

    # Dual to primal solution
    return x .= y-adjoint(g.A)*p

end

(g::WeightedProximableFun{T,N1,N2})(x::AbstractArray{T,N1}) where {T,N1,N2} = g.g(g.A*x)

Flux.gpu(g::WeightedProximableFun{T,N1,N2}) where {T,N1,N2} = WeightedProximableFun{T,N1,N2}(g.g, gpu(g.A))
Flux.cpu(g::WeightedProximableFun{T,N1,N2}) where {T,N1,N2} = WeightedProximableFun{T,N1,N2}(g.g, cpu(g.A))


# Proximable function evaluation


struct ProxyObjFun{T,N} <: DifferentiableFunction{T,N}
    λ::T
    g::ProximableFunction{T,N}
    opt::Union{Nothing,OptimOptions}
end

proxy_objfun(λ::T, g::ProximableFunction{T,N}) where {T,N} = ProxyObjFun{T,N}(λ, g, nothing)
proxy_objfun(λ::T, g::ProximableFunction{T,N}, opt::OptimOptions) where {T,N} = ProxyObjFun{T,N}(λ, g, opt)

function grad!(f::ProxyObjFun{T,N}, y::AbstractArray{T,N}, g::Union{Nothing,AbstractArray{T,N}}) where {T,N}
    f.opt === nothing ? (x = proxy(y, f.λ, f.g)) : (x = proxy(y, f.λ, f.g, f.opt))
    g !== nothing && (g .= y-x)
    return T(0.5)*norm(x-y)^2+f.λ*f.g(x)
end


struct ProjObjFun{T,N} <: DifferentiableFunction{T,N}
    ε::T
    g::ProximableFunction{T,N}
    opt::Union{Nothing,OptimOptions}
end

proj_objfun(ε::T, g::ProximableFunction{T,N}) where {T,N} = ProjObjFun{T,N}(ε, g, nothing)
proj_objfun(ε::T, g::ProximableFunction{T,N}, opt::OptimOptions) where {T,N} = ProjObjFun{T,N}(ε, g, opt)

function grad!(f::ProjObjFun{T,N}, y::AbstractArray{T,N}, g::Union{Nothing,AbstractArray{T,N}}) where {T,N}
    f.opt === nothing ? (x = project(y, f.ε, f.g)) : (x = project(y, f.ε, f.g, f.opt))
    g !== nothing && (g .= y-x)
    return T(0.5)*norm(x-y)^2
end


# Minimizable type utils

struct DiffPlusProxFun{T,N}<:MinimizableFunction{T,N}
    f::DifferentiableFunction{T,N}
    g::ProximableFunction{T,N}
end

Base.:+(f::DifferentiableFunction{T,N}, g::ProximableFunction{T,N}) where {T,N} = DiffPlusProxFun{T,N}(f, g)