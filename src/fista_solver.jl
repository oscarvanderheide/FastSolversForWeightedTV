#: Solver FISTA-like


export solverFISTA, optFISTA


# Options for FISTA

struct optFISTA{DT,T}
    initial_estimate::DT
    steplength::T
    niter::Int64
    tol::T
    log::Bool
    verbose::Bool
end

optFISTA(initial_estimate::DT, steplength::T; niter::Int64=100, tol::T=T(0), log::Bool=false, verbose::Bool=false) where {DT,T} = optFISTA{DT,T}(initial_estimate, steplength, niter, tol, log, verbose)


# Generic solver

"""
Solver for the regularized problem via FISTA-like gradient-projections:
```math
min_x f(x)+λ*g(x)
```
Reference: Beck, A., and Teboulle, M., 2009, A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems
https://www.ceremade.dauphine.fr/~carlier/FISTA
"""
function solverFISTA(f::DifferentiableFunction{DT}, λ::T, g::ProximableFunction{DT}, opt::optFISTA{DT,T}) where {DT<:AbstractField2D,T}

    # Initialization
    x0 = deepcopy(opt.initial_estimate)
    ∇f = initialize_as(x0) # gradient pre-allocation
    x = initialize_as(x0)  # dual-variable pre-allocation
    t0 = T(1)
    opt.log && (objective_hist = zeros(T, opt.niter))

    # Optimization loop
    for i = 1:opt.niter

        fval = grad!(∇f, f, x0)                      # Compute gradient
        gval = proxy!(x, λ, g, x0-opt.steplength*∇f) # Gradient update + projection
        if opt.nesterov                              # > Nesterov two-step update:
            t = T(0.5)*(T(1)+sqrt(T(1)+T(4)*t0^2))   # - compute dynamic step size
            update!(x, x+(t0-T(1))/t*(x-x0))         # - update momentum
            t0 = t                                   # - update step size
        end                                          # <
        update!(x0, x)                               # Update unknowns

        # Print iteration
        (opt.log || opt.verbose) && (objval = fval+gval)
        opt.log && (objective_hist[i] = objval)
        opt.verbose && println("Iter ", i, ", loss (primal): ", objval) 

    end

    # Return output
    opt.log ? (return objective_hist, x) : (return x)

end