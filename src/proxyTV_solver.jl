#: Proximal operator solvers for TV norm


export solverProxyTV, optProxyTV


# Options for ProxyL21

mutable struct optProxyTV <: Options
    initial_estimate
    steplength
    niter::Int64
    tol
    nesterov::Bool
    log::Bool
    verbose::Bool
end

optProxyTV(; steplength::T=T(1), niter::Int64=100, tol::T=T(0), nesterov::Bool=true, log::Bool=false, verbose::Bool=false) where T = optProxyTV(nothing, steplength, niter, tol, nesterov, log, verbose)


# Solver

"""
Solver for the TV-regularized problem:
```math
min_x 0.5*||x-y||^2+λ*||D*∇*x||_{2,1}
```
"""
function solverProxyTV(y::ScalarField2D{T}, λ::T; D::AbstractFieldLinearOperator{ScalarField2D{T},VectorField2D{T}}=IdentityFieldLinearOperator(VectorField2D{T}), opt::Options=optProxyTV()) where T

    # Initialization
    x0 = deepcopy(opt.initial_estimate)
    ∇f = undef_as(x0) # gradient pre-allocation
    x = undef_as(x0)  # dual-variable pre-allocation
    t0 = T(1)
    opt.log ? (objective_hist = zeros(T, opt.niter)) : (objective_hist = nothing)
    α = opt.steplength

    # Optimization loop
    for i = 1:opt.niter

        fval = grad!(∇f, f, x0)                    # Compute gradient
        gval = proxy!(x, α*λ, g, x0-α*∇f)          # Gradient update + projection
        if opt.nesterov                            # > Nesterov two-step update:
            t = T(0.5)*(T(1)+sqrt(T(1)+T(4)*t0^2)) # - compute dynamic step size
            update!(x, x+(t0-T(1))/t*(x-x0))       # - update momentum
            t0 = t                                 # - update step size
        end                                        # <
        check = norm(x-x0; p=Inf)/α
        check <= opt.tol && break
        update!(x0, x)                               # Update unknowns

        # Print iteration
        (opt.log || opt.verbose) && (objval = fval+gval)
        opt.log && (objective_hist[i] = objval)
        opt.verbose && println("Iter ", i, ", loss (primal): ", objval, ", check:", check) 

    end

    # Return output
    return objective_hist, x

end