#: Proximal operator solvers for {2,1} norm


export solverProxyL21, optProxyL21


# Options for ProxyL21

mutable struct optProxyL21 <: Options
    initial_estimate
    steplength
    niter::Int64
    tol::Union{Nothing,Real}
    nesterov::Bool
    log::Bool
    verbose::Bool
end

optProxyL21(; steplength::T=T(1), niter::Int64=100, tol::Union{Nothing,T}=nothing, nesterov::Bool=true, log::Bool=false, verbose::Bool=false) where T = optProxyL21(nothing, steplength, niter, tol, nesterov, log, verbose)


# Solver

"""
Solver for the regularized problem via gradient-projections:
```math
min_{x∈C} 0.5*||x-y||^2+λ*||L*x||_{2,1}
```
"""
function solverProxyL21(y::ScalarField2D{T}, λ::T, L::AbstractFieldLinearOperator{ScalarField2D{T},VectorField2D{T}}; C::ProjectionableSet=no_constraints(ScalarField2D{T}), opt::Options=optProxyL21()) where T

    # Initialization
    x  = undef_scalar_as(y) # primal-variable pre-allocation
    p0 = zeros_vector_as(y) # dual-variable pre-allocation
    q  = p0
    p  = undef_vector_as(y) # dual-variable pre-allocation
    t0 = T(1)
    opt.log ? (objective_hist = zeros(T, opt.niter)) : (objective_hist = nothing)
    α = opt.steplength

    # Constraints for dual set
    C_dual = ell_ball(T,2,Inf)

    # Optimization loop
    for i = 1:opt.niter

        x = project(y-λ*adjoint(L)*q, C)           # Solve for primal variable
        g = λ*L*x                                  # Compute dual gradient
        p = project(q+α*g, C_dual)                 # Gradient update + projection
        if opt.nesterov                            # > Nesterov two-step update:
            t = T(0.5)*(T(1)+sqrt(T(1)+T(4)*t0^2)) # - compute dynamic step size
            q = p+(t0-T(1))/t*(p-p0)               # - update momentum
            t0 = t                                 # - update step size
        else
            q = p
        end                                        # <
        (opt.tol isa Real) && (check = norm(p-p0; p1=2,p2=1)/α)
        p0 = p                                     # Update dual unknowns

        # Print iteration
        (opt.log || opt.verbose) && (objval = T(0.5)*norm(x-y)^2+norm(g; p1=2, p2=1))
        opt.log && (objective_hist[i] = objval)
        opt.verbose && println("Iter ", i, ", loss (primal): ", objval, ", check:", check)
        (opt.tol isa Real) && (check <= opt.tol && break)

    end

    # Return output
    return objective_hist, x

end