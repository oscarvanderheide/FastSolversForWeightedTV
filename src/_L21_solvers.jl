#: Solver for least-squares problem regularized by l_{2,1} norm


export solverProxyNorm21, solverLSNorm21


# Solvers

"""
Proximal operator solver
```math
\min_p \dfrac{1}{2}||p-q||_{2,2}^2+λ||q||_{2,1}
```
"""
function solverProxyNorm21(p::VectorField{T}, λ::T) where T
    pt_normp = ptnorm(p)
    return (T(1)-λ/pt_normp)*p*(pt_normp >= λ), pt_normp
end
function solverProxyNorm21!(q::VectorField{T}, pt_normp::ScalarField2D{T}, p::VectorField{T}, λ::T) where T
    pt_normp .= ptnorm(p)
    q.array .= (T(1)-λ/pt_normp.array)*p.array*(pt_normp.array .>= λ)
end

"""
Solver for the regularized least-squares problem via FISTA-like gradient-projections:
```math
\min_q \dfrac{1}{2}||Lq-p||_{2,2}^2+λ||q||_{2,1}
```
Reference: Beck, A., and Teboulle, M., 2009, A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems
https://www.ceremade.dauphine.fr/~carlier/FISTA
"""
function solverLSNorm21(L::FieldLinearOperator{T}, p::VectorField2D{T}, λ::T; opt::optSolver{T} = optLSNorm21()) where T

    # Initialization
    q0 = opt.initial_estimate # type: VectorField2D{T}
    normq0 = initialize_as(q0; type="scalar")
    g0 = initialize_as(q0)
    q  = initialize_as(q0)
    t0 = T(1)
    opt.log && (fval = zeros(T, opt.niter))

    # Setting proximal operator for ||q||_{2,1}
    proxy!(q, normq, p) = solverProxyNorm21!(q, normq, p, λ)

    # Optimization loop
    for i = 1:opt.niter

        r = L*q0-p                                 # Computing residual
        update!(g0, adjoint(L)*r)                  # Compute gradient
        proxy!(q, normq0, q0-opt.steplength*g0)    # Gradient update + projection

        (opt.log || opt.verbose) && (f = T(0.5)*norm(r)^2+λ*normq0) # primal loss
        opt.log && (fval[i] = f)                                    # store log
        pt.verbose && println("Iter ", i, ", loss (primal): ", f)   # print

        if opt.nesterov                            # Nesterov two-step update:
            t = T(0.5)*(T(1)+sqrt(T(1)+T(4)*t0^2)) # - compute dynamic step size
            update!(q, q+(t0-T(1))/t*(q-q0))       # - update momentum
            t0 = t                                 # - update step size
        end
        update!(q0, q)                             # Update unknowns

    end

    # Return output
    opt.log ? (return q, fval) : (return q)

end

update!(v_::VectorField2D{T}, v::VectorField2D{T}) where T = (v_.array .= v.array)