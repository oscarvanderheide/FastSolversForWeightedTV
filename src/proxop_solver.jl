# Proximal operator solvers


export solverTVFastGradientProjection, optTVFGP


## Option data type

mutable struct optTVFGP
    niter::Int64
    s::Union{Nothing, T}
    D::Union{Nothing, LinearMap{T}}
    Proj::Union{Nothing, Function}
    verbose::Bool
    log::Bool
end

optTVFGP(; niter::Int64=100, s::Union{Nothing,T}=nothing, D::Union{Nothing,LinearMap{T}}=nothing, Proj::Union{Nothing,Function}=nothing, verbose::Bool=false, log::Bool=false) = optTVFGP(niter, s, D, Proj, verbose, log)


## Solver

function solverTVFastGradientProjection(y::AbstractArray{T,2}, α::T; h::Tuple{T,T}=(T(1),T(1)), opt::optTVFGP = optTVFGP())

    # Filling-in missing options
    n = size(y)
    isnothing(opt.D) ? (D = identityMapVectorField2D(n)) : (D = opt.D)
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
    p0 = VectorField2D(cuzeros(y,n), cuzeros(y,n))
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