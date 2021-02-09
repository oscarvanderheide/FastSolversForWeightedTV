#: Optimization functions and solvers

export DiffPlusProxFunction, solverFISTA, optFISTA


# Optimization type

struct DiffPlusProxFunction{DT}
    f::DifferentiableFunction{DT}
    g::ProximableFunction{DT}
end

import Base: +
+(f::DifferentiableFunction{DT}, g::ProximableFunction{DT}) where DT = DiffPlusProxFunction{DT}(f, g)


# Options type for FISTA

abstract type Options end

mutable struct optFISTA<:Options
    initial_estimate::AbstractArray
    steplength::Number
    niter::Int64
    tol_obj::Number
    nesterov::Bool
    log::Bool
    verbose::Bool
end

optFISTA(initial_estimate::AbstractArray{T,N}; steplength::T=T(1), niter::Int64=100, tol_obj::T=T(0), nesterov::Bool=true, log::Bool=false, verbose::Bool=false) where {T,N} = optFISTA(initial_estimate, steplength, niter, tol_obj, nesterov, log, verbose)


# Generic FISTA solver

"""
Solver for the regularized problem via FISTA-like gradient-projections:
```math
min_x f(x)+g(x)
```
Reference: Beck, A., and Teboulle, M., 2009, A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems
https://www.ceremade.dauphine.fr/~carlier/FISTA
"""
function solverFISTA(fun::DiffPlusProxFunction{DT}, opt::Options) where {T,N,DT<:AbstractArray{T,N}}

    # Initialization
    x0 = copy(opt.initial_estimate)
    ∇f = similar(x0)                             # gradient pre-allocation
    opt.nesterov && (x = similar(x0); t0 = T(1)) # momentum pre-allocation
    opt.log ? (objective_hist = zeros(T, opt.niter)) : (objective_hist = nothing)
    objval0 = T(0)

    # Optimization loop
    for i = 1:opt.niter

        fval = grad!(fun.f, x0, ∇f)                              # Compute gradient
        if opt.nesterov                                          # > Nesterov two-step update:
            gval = proxy!(x0-opt.steplength*∇f, T(1), fun.g, x)  # Compute proxy
            t = T(0.5)*(T(1)+sqrt(T(1)+T(4)*t0^2))               # - compute dynamic step size
            x .= x+(t0-T(1))/t*(x-x0)                            # - update momentum
            t0 = t                                               # - update step size
            x0 .= x                                              # Update unknowns
        else                                                     #
            gval = proxy!(x0-opt.steplength*∇f, T(1), fun.g, x0) # Compute proxy
        end                                                      # <

        # Print iteration
        objval = fval+gval
        opt.log && (objective_hist[i] = objval)
        opt.verbose && println("Iter ", i, ", loss: ", objval)

        # Convergence check
        i > 1 && abs(objval-objval0)/abs(objval0) <= opt.tol_obj && break
        objval0 = objval

    end

    # Return output
    return x, objective_hist

end