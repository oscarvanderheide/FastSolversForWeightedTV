#: Utilities for norm functions

export ptdot_2D, ptnorm1_2D, ptnorm2_2D, ptnormInf_2D, normTV_2D
export l22_norm_2D, l21_norm_2D, l2Inf_norm_2D


# Concrete norm types

struct MixedNorm_2D{T,N1,N2}<:ProximableFunction{T,3} end

struct WeightedMixedNorm_2D{T,N1,N2,WT<:AbstractLinearOperator}<:ProximableFunction{T,3}
    weight::WT
    solver_opt::OptimOptions
end

norm_2D(N1::Int64, N2::Int64; T::DataType=Float32) = MixedNorm_2D{T,N1,N2}()


# L22 norm

function proxy!(p::AbstractArray{T,3}, λ::T, ::MixedNorm_2D{T,2,2}, q::AbstractArray{T,3}) where T
    np = norm(p)
    np <= λ ? (return q .= T(0)) : (return q .= (T(1)-λ/np)*p)
end

function project!(p::AbstractArray{T,3}, ε::T, ::MixedNorm_2D{T,2,2}, q::AbstractArray{T,3}) where T
    np = norm(p)
    np <= ε ? (return q .= p) : (return q .= ε*p/np)
end

(::MixedNorm_2D{T,2,2})(p::AbstractArray{T,3}) where T = norm(p)


# L21 norm

function proxy!(p::AbstractArray{T,3}, λ::T, ::MixedNorm_2D{T,2,1}, q::AbstractArray{T,3}; ptn::Union{AbstractArray{T,2},Nothing}=nothing) where T
    ptn === nothing && (ptn = ptnorm2_2D(p; η=eps(T)))
    return q .= (T(1).-λ./ptn).*(ptn .>= λ).*p
end

function project!(p::AbstractArray{T,3}, ε::T, g::MixedNorm_2D{T,2,1}, q::AbstractArray{T,3}) where T
    ptn = ptnorm2_2D(p)
    sum(ptn) <= ε && (return q .= p)
    λ = pareto_search_proj21(ptn, ε)
    return proxy!(p, λ, g, q; ptn=ptn)
end

function pareto_search_proj21(ptn::AbstractArray{T,2}, ε::T) where T
    ptn = sort(vec(ptn))
    Σptn = cumsum(ptn[end:-1:1])[end:-1:1]-(length(ptn):-1:1).*ptn
    val = Σptn .<= ε
    i = length(val)-sum(val)
    CUDA.@allowscalar return ptn[i]+(ε-Σptn[i])/(Σptn[i+1]-Σptn[i])*(ptn[i+1]-ptn[i])
end

(::MixedNorm_2D{T,2,1})(p::AbstractArray{T,3}) where T = norm21_2D(p)


# L2Inf norm

function proxy!(p::AbstractArray{T,3}, λ::T, ::MixedNorm_2D{T,2,Inf}, q::AbstractArray{T,3}) where T
    project!(p, λ, l21_norm_2D(; T=T), q)
    return q .= p.-q
end

function project!(p::AbstractArray{T,3}, ε::T, ::MixedNorm_2D{T,2,Inf}, q::AbstractArray{T,3}) where T
    ptn = ptnorm2_2D(p; η=eps(T))
    val = ptn .>= ε
    q .= p.*(ε*val./ptn+(!).(val))
    return q
end

(::MixedNorm_2D{T,2,Inf})(p::AbstractArray{T,3}) where T = norm2Inf_2D(p)


# Weighted norm

function proxy!(y::AbstractArray{T,2}, λ::T, g::WeightedMixedNorm_2D{T,N1,N2}, x::AbstractArray{T,2}) where {T,N1,N2}

    # Objective function (dual problem)
    f = leastsquares_misfit(λ*adjoint(g.weight), y)+λ*conjugate(norm_2D(N1, N2; T=T))

    # Minimization (dual variable)
    p = similar(y, size(y)..., 3); p .= T(0)
    p = minimize(f, p, g.solver_opt)

    # Dual to primal solution
    x .= y-λ*adjoint(g.A)*p

    return x

end

function project!(p::AbstractArray{T,3}, ε::T, ::MixedNorm_2D{T,2,2}, q::AbstractArray{T,3}) where T
    np = norm(p)
    np <= ε ? (return q .= p) : (return q .= ε*p/np)
end

(::MixedNorm_2D{T,2,2})(p::AbstractArray{T,3}) where T = norm(p)


# Utils

ptdot_2D(v1::AbstractArray{T,3}, v2::AbstractArray{T,3}) where T = sum(v1.*v2; dims=3)[:,:,1]
ptnorm1_2D(p::AbstractArray{T,3}; η::T=T(0)) where T = abs.(p[:,:,1])+abs.(p[:,:,2]).+η
ptnorm2_2D(p::AbstractArray{T,3}; η::T=T(0)) where T = sqrt.(p[:,:,1].^T(2)+p[:,:,2].^T(2).+η^2)
ptnormInf_2D(p::AbstractArray{T,3}; η::T=T(0)) where T = maximum(abs.(p).+η; dims=3)[:,:,1]

norm21_2D(v::AbstractArray{T,3}; η::T=T(0)) where T = sum(ptnorm2_2D(v; η=η))
norm22_2D(v::AbstractArray{T,3}; η::T=T(0)) where T = sqrt(sum(ptnorm2_2D(v; η=η).^2))
norm2Inf_2D(v::AbstractArray{T,3}; η::T=T(0)) where T = maximum(ptnorm2_2D(v; η=η))