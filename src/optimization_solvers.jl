#: Optimization functions and solvers

export OptFISTA, opt_fista


# Options type for FISTA

mutable struct OptFISTA{T<:Real}<:OptimOptions
    steplength::T
    niter::Int64
    tol_x::Union{Nothing,T}
    nesterov::Bool
end

opt_fista(steplength::T; niter::Int64=1000, tol_x::Union{Nothing,T}=nothing, nesterov::Bool=true) where {T<:Real} = OptFISTA{T}(steplength, niter, tol_x, nesterov)


# Generic FISTA solver

"""
Solver for the regularized problem via FISTA-like gradient-projections:
```math
min_x f(x)+g(x)
```
Reference: Beck, A., and Teboulle, M., 2009, A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems
https://www.ceremade.dauphine.fr/~carlier/FISTA
"""
function minimize_fista!(fun::DiffPlusProxFun{CT,N}, initial_estimate::AbstractArray{CT,N}, steplength::T, niter::Int64, nesterov::Bool, tol_x::Union{Nothing,T}, x0::AbstractArray{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}}

    # Initialization
    x0   .= initial_estimate
    xtmp  = similar(x0)
    x     = similar(x0)
    t0 = T(1)

    # Optimization loop
    for i = 1:niter

        grad!(fun.f, x0, xtmp; eval=false)         # Compute gradient
        xtmp .= x0-steplength*xtmp               # Gradient update
        # xtmp .*= -steplength; xtmp .+= x0          # Gradient update
        proxy!(xtmp, steplength, fun.g, x)         # Compute proxy
        if nesterov                                # > Nesterov two-step update:
            t = T(0.5)*(T(1)+sqrt(T(1)+T(4)*t0^2)) # - compute dynamic step size
            x .+= (t0-T(1))/t*(x-x0)               # - update momentum
            # x .*= (t+t0-T(1))/t; x .-= (t0-T(1))/t*x0 # - update momentum
            t0 = t                                 # - update step size
        end                                        # <

        # Update unknowns
        ~isnothing(tol_x) && norm(x-x0)/norm(x0) < tol_x && break
        x0 .= x

    end

    # Return output
    return x0

end

function minimize_fista_debug!(fun::DiffPlusProxFun{CT,N}, initial_estimate::AbstractArray{CT,N}, steplength::T, niter::Int64, nesterov::Bool, tol_x::Union{Nothing,T}, x0::AbstractArray{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}}

    # Initialization
    x0 .= initial_estimate
    xtmp = similar(x0)
    x = similar(x0)
    t0 = T(1)
    fval_hist = Array{T,1}(undef, niter)
    err_rel = Array{T,1}(undef, niter)

    # Optimization loop
    for i = 1:niter

        fval = grad!(fun.f, x0, xtmp; eval=true)   # Compute gradient
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
        ~isnothing(tol_x) && err_rel[i] < tol_x && break

        # Update unknowns
        x0 .= x

    end

    # Return output
    return x0, fval_hist, err_rel

end

minimize!(fun::DiffPlusProxFun{CT,N}, x0::AbstractArray{CT,N}, opt::OptFISTA{T}, x::AbstractArray{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}} = minimize_fista!(fun, x0, opt.steplength, opt.niter, opt.nesterov, opt.tol_x, x)

minimize_debug!(fun::DiffPlusProxFun{CT,N}, x0::AbstractArray{CT,N}, opt::OptFISTA{T}, x::AbstractArray{CT,N}) where {T<:Real,N,CT<:RealOrComplex{T}} = minimize_fista_debug!(fun, x0, opt.steplength, opt.niter, opt.nesterov, opt.tol_x, x)