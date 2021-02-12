#: Proximable function utilities


export ℓnorm_2D, ell_norm, elastic_net_vect


# ℓnorm_2D

struct ℓnorm_2D{T,p1,p2}<:ProximableFunction{T,3} end

ell_norm(T::DataType, p1::Number, p2::Number) = ℓnorm_2D{T,p1,p2}()

(g::ℓnorm_2D{T,2,1})(x::AbstractArray{T,3}) where T = norm21(x)
(g::ℓnorm_2D{T,2,2})(x::AbstractArray{T,3}) where T = norm22(x)
(g::ℓnorm_2D{T,2,Inf})(x::AbstractArray{T,3}) where T = norm2Inf(x)


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

### Utils for root-finding 2,1

function pareto_search_proj21(ptn::AbstractArray{T,2}) where T
    obj_fun = δ->objfun_paretosearch_proj21(δ, ptn)
    return find_zero(obj_fun, (T(0), maximum(ptn)))
end

function objfun_paretosearch_proj21(δ::T, ptn::AbstractArray{T,2}) where T
    ptn_ = ptn[ptn.>=δ]
    return T(1)-sum(ptn_)+length(ptn_)*δ
end


# 2, 2

function proxy!(p::DT, λ::T, ::ℓnorm_2D{T,2,2}, q::DT) where {T,DT<:AbstractArray{T,3}}
    n = norm22(p)
    q .= (T(1)-λ/n)*(n>=λ)*p
    return q
end

function project!(p::DT, ε::T, ::ℓnorm_2D{T,2,2}, q::DT) where {T,DT<:AbstractArray{T,3}}
    q .= ε*p/norm22(p)
    return q
end


# 2, Inf

function proxy!(p::DT, λ::T, ::ℓnorm_2D{T,2,Inf}, q::DT) where {T,DT<:AbstractArray{T,3}}
    C = ell_ball(2, 1, T(1))
    project!(p/λ, C, q)
    q .= p-λ*q
    return q
end

function project!(p::DT, ε::T, ::ℓnorm_2D{T,2,Inf}, q::DT) where {T,DT<:AbstractArray{T,3}}
    ptn = ptnorm2(p)
    q .= p.*((ptn.>=ε)./ptn+(ptn.<ε))
    return q
end


# Elastic net

struct ElasticNetVect_2D{T}<:ProximableFunction{T,3}
    μ::T
end

(g::ElasticNetVect_2D{T})(x::AbstractArray{T,3}) where T = norm21(x)+g.μ^2/T(2)*norm22(x)^2

function proxy!(p::DT, λ::T, g::ElasticNetVect_2D{T}, q::DT; ptn::Union{RT,Nothing}=nothing) where {T,DT<:AbstractArray{T,3},RT<:AbstractArray{T,2}}
    ptn === nothing && (ptn = ptnorm2(p))
    q .= T(1)/(T(1)+g.μ^2*λ)*(T(1).-λ./ptn).*(ptn .>= λ).*p
end

elastic_net_vect(μ::T) where T = ElasticNetVect_2D{T}(μ)