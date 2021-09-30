#: Abstract functional types

export MinimizableFunction, DifferentiableFunction, ProximableFunction, ProjectionableSet
export minimize, minimize!, proxy, proxy!, project, project!, grad, grad!


# Abstract type declarations

"""Expected behavior for MinimizableFunction:
- minimize!(f::MinimizableFunction{T,N}, x0::AbstractArray{T,N}) where {T,N}
It approximates the solution of the optimization problem: min_x f(x)
"""
abstract type MinimizableFunction{T,N} end

minimize(f::MinimizableFunction{T,N}, x0::AbstractArray{T,N}) where {T,N} = minimize!(f, deepcopy(x0))

"""Expected behavior for DifferentiableFunction:
- fval::T = grad!(f::DifferentiableFunction{DT}, x::DT, g::DT) where {T,N,DT<:AbstractArray{T,N}} 
"""
abstract type DifferentiableFunction{T,N} end

function grad(f::DifferentiableFunction{T,N}, x::AbstractArray{T,N}) where {T,N}
    g = similar(x)
    fval = grad!(f, x, g)
    return fval, g
end

(f::DifferentiableFunction{T,N})(x::AbstractArray{T,N}) where {T,N} = grad!(f, x, nothing)

"""Expected behavior for ProximableFunction:
- proxy!(y::AbstractArray{T,N}, λ::T, g::ProximableFunction{T,N}, x::AbstractArray{T,N}) where {T,N}
- proxy!(y::AbstractArray{T,N}, λ::T, g::ProximableFunction{T,N}, x::AbstractArray{T,N}, opt::OptimOptions) where {T,N}
It approximates the solution of the optimization problem: min_x 0.5*norm(x-y)^2+λ*g(x)
- project!(y::AbstractArray{T,N}, ε::T, g::ProximableFunction{T,N}, x::AbstractArray{T,N}) where {T,N}
- project!(y::AbstractArray{T,N}, ε::T, g::ProximableFunction{T,N}, x::AbstractArray{T,N}, opt::OptimOptions) where {T,N}
It approximates the solution of the optimization problem: min_{g(x)<=ε} 0.5*norm(x-y)^2
"""
abstract type ProximableFunction{T,N} end

proxy(y::AbstractArray{T,N}, λ::T, g::ProximableFunction{T,N}) where {T,N} = proxy!(y, λ, g, similar(y))
project(y::AbstractArray{T,N}, ε::T, g::ProximableFunction{T,N}) where {T,N} = project!(y, ε, g, similar(y))
proxy(y::AbstractArray{T,N}, λ::T, g::ProximableFunction{T,N}, opt::OptimOptions) where {T,N} = proxy!(y, λ, g, similar(y), opt::OptimOptions)
project(y::AbstractArray{T,N}, ε::T, g::ProximableFunction{T,N}, opt::OptimOptions) where {T,N} = project!(y, ε, g, similar(y), opt::OptimOptions)

export ProjectionableSet, SubLevelSet, no_constraints, indicator


"""Projectional sets
Expected behavior for convex sets: y = project!(x, C, y), y = Π_C(x)
"""
abstract type ProjectionableSet{T,N} end

project(x::AbstractArray{T,N}, C::ProjectionableSet{T,N}) where {T,N} = project!(x, C, similar(x))
project(x::AbstractArray{T,N}, C::ProjectionableSet{T,N}, opt::OptimOptions) where {T,N} = project!(x, C, similar(x), opt)