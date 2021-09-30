#: Proximable function utilities


export ℓnorm_2D, ell_norm, elastic_net_vect, tv_norm_2D


# 2, 1

function proxy!(p::DT, λ::T, g::ℓnorm_2D{T,2,1}, q::DT; ptn::Union{RT,Nothing}=nothing) where {T,DT<:AbstractArray{T,3},RT<:AbstractArray{T,2}}
    ptn === nothing && (ptn = ptnorm2(p))
    q .= (T(1).-λ./ptn).*(ptn .>= λ).*p
    return q
end

function project!(p::DT, ε::T, g::ℓnorm_2D{T,2,1}, q::DT) where {T,DT<:AbstractArray{T,3}}
    q .= p/ε
    ptn = ptnorm2(q)
    λ = pareto_search_proj21(ptn)
    proxy!(q, λ, g, q; ptn=ptn)
    q .= ε*q
    return q
end


# A-ell_norm

struct Aℓnorm_2D{T,p1,p2}<:ProximableFunction{T,2}
    A::AbstractLinearOperator
    opt::OptimOptions
end

(g::Aℓnorm_2D{T,2,1})(x::AbstractArray{T,2}) where T = norm21(g.A*x)

function proxy!(y::DT, λ::T, g::Aℓnorm_2D{T,2,1}, x::DT) where {T,DT<:AbstractArray{T,2}}

    # Minimization function
    f = leastsquares_misfit(λ*adjoint(g.A), y)+indicator(ell_norm(T, 2, Inf), T(1))

    # Minimization (dual variable)
    y isa CuArray ? (p0 = CUDA.zeros(T, size(y)..., 2)) : (p0 = zeros(T, size(y)..., 2))
    p = minimize(f, p0, g.opt)

    # Dual to primal solution
    x .= y-λ*adjoint(g.A)*p

    return x

end


# TV-related norm

function tv_norm_2D(T::DataType, n::Tuple{Int64,Int64}; h::NTuple{2}=(1f0,1f0), gpu::Bool=false, opt::Union{Nothing,OptimOptions}=nothing)
    isnothing(opt) && (opt = opt_fista(; steplength=T(1/8), niter=10))
    isnothing(opt.steplength) && (opt.steplength = T(1/8))
    return Aℓnorm_2D{T,2,1}(gradient_2D(n; h=T.(h), gpu=gpu), opt)
end