#: Proximal operator solvers


export solverProxyNorm21, solverProxyTV, optProxyTV


## Option data type

abstract type optSolver{T} end

struct optProxyTV{T} <: optSolver{T}
    steplength_primal::T
    steplength_dual  ::T
    niter            ::Int64
    s                ::T
    D                ::FieldLinearOperator{T}
    Π_primal         ::FieldLinearOperator{T}
    verbose          ::Bool
    log              ::Bool
end

optProxyTV{T}(; steplength_primal::T=T(1), steplength_dual::Tniter::Int64=100, s::Union{Nothing,T}=nothing, D::Union{Nothing,LinearMap{T}}=nothing, Proj::Union{Nothing,Function}=nothing, verbose::Bool=false, log::Bool=false) where T = optTVFGP(niter, s, D, Proj, verbose, log)


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
    normq0 = ScalarField2D{T}(typeof(q0.array)(undef, size(q0)[1:4]))
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


"""
Proximal operator solver
```math
\min_x \dfrac{1}{2}||x-y||^2+λ||D\,∇\,x||_{2,1},\mathrm{ where }L:\mathbb{R}^n\to\mathbb{R}^{2n}
```
Reference: Beck, A., and Teboulle, M., 2009, A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems
https://www.ceremade.dauphine.fr/~carlier/FISTA
"""
function solverProxyTV(y::ScalarField{T}, λ::T; opt::optProxyTV = optProxyTV()) where T

    # Filling-in missing options
    n = size(y)
    D = opt.D
    isnothing(opt.s) ? (s = T(1)/(T(8)*α^2)) : (s = opt.s)
    niter = opt.niter
    verbose = opt.verbose
    log = opt.log

    # Setting primal/dual projections
    isnothing(opt.Proj) ? (Proj(x) = x) : (Proj = opt.Proj)
    Proj_dual(p) = ptProj_L2(p)

    # Differential operator for TV
    ∇ = gradient_op(n; h=h, flag_gpu=use_gpu(y))
    div = -adjoint(∇)

    # Minimization loop
    log && (fval = zeros(T, niter))
    p0 = VectorField2D(zeros_as(y,n), zeros_as(y,n))
    q = p0
    t0 = T(1)
    for i = 1:niter
        u = Proj(y+α*div*(adjoint(D)*q))       # update primal variable
        ∇u = ∇*u; g = α*D*∇u                   # compute gradient
        p = Proj_dual(q+s*g)                   # update dual variable
        t = T(0.5)*(T(1)+sqrt(T(1)+T(4)*t0^2)) # update step size
        q = p+(t0-T(1))/t*(p-p0)               # Nesterov two-step update
        p0 = p
        t0 = t

        (log || verbose) && (f = 0.5*norm(y-u)^2+α*norm(∇u; p1=1, p2=2)) # primal loss
        log && (fval[i] = f)                                             # store log
        verbose && println("Iter ", i, ", loss (primal): ", f)           # print
    end

    # Final primal variable computation
    u = Proj(y+α*div*(adjoint(D)*q))

    log ? (return u, fval) : (return u)

end

ptProj_L2(p::VectorField2D) = p/max.(T(1), ptnorm(p; p=2)) # Pointwise L2 projection of a vector field