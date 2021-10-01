#: Utilities for norm functions

export ptdot_2D, ptnorm1_2D, ptnorm2_2D, ptnormInf_2D
export norm_2D, TV_norm_2D


# Concrete norm types

struct MixedNorm_2D{T,N1,N2}<:ProximableFunction{T,3} end

struct WeightedMixedNorm_2D{T,N1,N2,WT<:AbstractLinearOperator}<:ProximableFunction{T,3}
    weight::WT
    solver_opt::OptimOptions
end

norm_2D(N1::Number, N2::Number; T::DataType=Float32) = MixedNorm_2D{T,N1,N2}()


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

pareto_search_proj21(ptn::AbstractArray{T,2}, ε::T) where T = T(solve(ZeroProblem(λ -> obj_pareto_search_proj21(λ, ptn, ε), T(0)), Order1()))

obj_pareto_search_proj21(λ::T, ptn::AbstractArray{T,2}, ε::T) where T = sum(Flux.relu.(ptn.-λ))-ε

# function pareto_search_proj21(ptn::AbstractArray{T,2}, ε::T) where T
#     f = λ -> obj_pareto_search_proj21_Newton(λ, ptn, ε)
#     return T(solve(ZeroProblem(f, T(0)), Roots.Newton()))
# end

# function obj_pareto_search_proj21_Newton(λ::T, ptn::AbstractArray{T,2}, ε::T) where T
#     dp = ptn.-λ
#     f  =  sum(Flux.relu.(dp))-ε
#     df = -sum(∇relu(dp))
#     return (f, f/df)
# end

# ∇relu(x::AbstractArray{T,N}) where {T,N} = (sign.(x).+T(1))./T(2)

(::MixedNorm_2D{T,2,1})(p::AbstractArray{T,3}) where T = norm21_2D(p)


# L2Inf norm

function proxy!(p::AbstractArray{T,3}, λ::T, ::MixedNorm_2D{T,2,Inf}, q::AbstractArray{T,3}) where T
    project!(p, λ, norm_2D(2,1; T=T), q)
    return q .= p.-q
end

function project!(p::AbstractArray{T,3}, ε::T, ::MixedNorm_2D{T,2,Inf}, q::AbstractArray{T,3}) where T
    ptn = ptnorm2_2D(p; η=eps(T))
    val = ptn .>= ε
    q .= p.*(ε*val./ptn+(!).(val))
    return q
end

(::MixedNorm_2D{T,2,Inf})(p::AbstractArray{T,3}) where T = norm2Inf_2D(p)


# TV norm

function TV_norm_2D(; T::DataType=Float32, h::Tuple{S,S}=(T(1),T(1)), weight::Union{Nothing,AbstractLinearOperator}=nothing) where {S<:Number}
    ∇ = gradient_2D(; T=T, h=h)
    weight !== nothing ? (A∇ = weight*∇) : (A∇ = ∇)
    return MixedNorm_2D{T,2,1}()∘A∇
end


# Utils

ptdot_2D(v1::AbstractArray{T,3}, v2::AbstractArray{T,3}) where T = sum(v1.*v2; dims=3)[:,:,1]
ptnorm1_2D(p::AbstractArray{T,3}; η::T=T(0)) where T = abs.(p[:,:,1])+abs.(p[:,:,2]).+η
ptnorm2_2D(p::AbstractArray{T,3}; η::T=T(0)) where T = sqrt.(p[:,:,1].^T(2)+p[:,:,2].^T(2).+η^2)
ptnormInf_2D(p::AbstractArray{T,3}; η::T=T(0)) where T = maximum(abs.(p).+η; dims=3)[:,:,1]

norm21_2D(v::AbstractArray{T,3}; η::T=T(0)) where T = sum(ptnorm2_2D(v; η=η))
norm22_2D(v::AbstractArray{T,3}; η::T=T(0)) where T = sqrt(sum(ptnorm2_2D(v; η=η).^2))
norm2Inf_2D(v::AbstractArray{T,3}; η::T=T(0)) where T = maximum(ptnorm2_2D(v; η=η))