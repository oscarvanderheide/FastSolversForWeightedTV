#: Optimization functions and solvers

export OptimOptions, minimize, minimize!


# Optimization abstract types

abstract type OptimOptions end

minimize(fun::MinimizableFunction{T,N}, x0::AbstractArray{T,N}, opt::OptimOptions) where {T,N} = minimize!(fun, x0, opt, similar(x0))
minimize_debug(fun::MinimizableFunction{T,N}, x0::AbstractArray{T,N}, opt::OptimOptions) where {T,N} = minimize_debug!(fun, x0, opt, similar(x0))