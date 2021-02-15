#: Optimization functions and solvers

export OptimOptions, OptFISTA, opt_fista, minimize, minimize!, minimize_fista!


# Optimization abstract type

abstract type OptimOptions{T} end


# Options type for FISTA

mutable struct OptFISTA{T}<:OptimOptions{T}
    initial_estimate::AbstractArray{T}
    steplength::T
    niter::Int64
    tol_x::Union{Nothing,T}
    nesterov::Bool
end

opt_fista(; initial_estimate::AbstractArray{T,N}, steplength::T=T(1), niter::Int64=1000, tol_x::Union{Nothing,T}=nothing, nesterov::Bool=true) where {T,N} = OptFISTA{T}(initial_estimate, steplength, niter, tol_x, nesterov)


# Generic FISTA solver

"""
Solver for the regularized problem via FISTA-like gradient-projections:
```math
min_x f(x)+g(x)
```
Reference: Beck, A., and Teboulle, M., 2009, A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems
https://www.ceremade.dauphine.fr/~carlier/FISTA
"""
function minimize_fista!(fun::DiffPlusProxFunction{T,N}, initial_estimate::DT, steplength::T, niter::Int64, nesterov::Bool, tol_x::Union{Nothing,T}, x0::DT) where {T,N,DT<:AbstractArray{T,N}}

    # Initialization
    x0 .= initial_estimate
    xtmp = similar(x0)
    x = similar(x0)
    t0 = T(1)

    # Optimization loop
    for i = 1:niter

        fval = grad!(fun.f, x0, xtmp)              # Compute gradient
        xtmp .= x0-steplength*xtmp                 # Gradient update
        proxy!(xtmp, steplength, fun.g, x)         # Compute proxy
        if nesterov                                # > Nesterov two-step update:
            t = T(0.5)*(T(1)+sqrt(T(1)+T(4)*t0^2)) # - compute dynamic step size
            x .= x+(t0-T(1))/t*(x-x0)              # - update momentum
            t0 = t                                 # - update step size
        end                                        # <

        # Update unknowns
        x0 .= x

    end

    # Return output
    return x0

end

function minimize_fista_debug!(fun::DiffPlusProxFunction{T,N}, initial_estimate::DT, steplength::T, niter::Int64, nesterov::Bool, x0::DT) where {T,N,DT<:AbstractArray{T,N}}

    # Initialization
    x0 .= initial_estimate
    xtmp = similar(x0)
    x = similar(x0)
    t0 = T(1)
    fval_hist = Array{T,1}(undef, niter)
    err_rel = Array{T,1}(undef, niter)

    # Optimization loop
    for i = 1:niter

        fval = grad!(fun.f, x0, xtmp)              # Compute gradient
        xtmp .= x0-steplength*xtmp                 # Gradient update
        proxy!(xtmp, steplength, fun.g, x)         # Compute proxy
        if nesterov                                # > Nesterov two-step update:
            t = T(0.5)*(T(1)+sqrt(T(1)+T(4)*t0^2)) # - compute dynamic step size
            x .= x+(t0-T(1))/t*(x-x0)              # - update momentum
            t0 = t                                 # - update step size
        end                                        # <

        # Convergence check
        fval_hist[i] = fval
        err_rel[i] = norm(x-x0)/norm(x0)

        # Update unknowns
        x0 .= x

    end

    # Return output
    return x0, fval_hist, err_rel

end

minimize!(fun::DiffPlusProxFunction{T,N}, opt::OptFISTA{T}, x::AbstractArray{T,N}) where {T,N} = minimize_fista!(fun, opt.initial_estimate, opt.steplength, opt.niter, opt.nesterov, opt.tol_x, x)

minimize_debug!(fun::DiffPlusProxFunction{T,N}, opt::OptFISTA{T}, x::AbstractArray{T,N}) where {T,N} = minimize_fista_debug!(fun, opt.initial_estimate, opt.steplength, opt.niter, opt.nesterov, x)

minimize(fun::OptimizableFunction{T,N}, opt::OptimOptions{T}) where {T,N} = minimize!(fun, opt, similar(opt.initial_estimate))

minimize_debug(fun::OptimizableFunction{T,N}, opt::OptimOptions{T}) where {T,N} = minimize_debug!(fun, opt, similar(opt.initial_estimate))