#: Utilities for norm functions

export ptdot, ptnorm1, ptnorm2, ptnormInf
export norm_2D, norm_batch_2D, L2V_norm_2D, LInfV_norm_2D, TV_norm_2D, TV_norm_batch_2D
export norm_3D, norm_batch_3D, L2V_norm_3D, LInfV_norm_3D, TV_norm_3D, TV_norm_batch_3D


# Mixed norm

struct MixedNorm_2D{T,N1,N2}<:ProximableFunction{T,3} end

struct WeightedMixedNorm_2D{T,N1,N2,WT<:AbstractLinearOperator}<:ProximableFunction{T,3}
    weight::WT
    solver_opt::OptimOptions
end

norm_2D(N1::Number, N2::Number; T::DataType=Float32) = MixedNorm_2D{T,N1,N2}()


# Mixed norm (batch)

struct MixedNormBatch_2D{T,N1,N2}<:ProximableFunction{T,4} end

norm_batch_2D(N1::Number, N2::Number; T::DataType=Float32) = MixedNormBatch_2D{T,N1,N2}()


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


# L21 norm (batch)

function proxy!(p::AbstractArray{T,4}, λ::T, ::MixedNormBatch_2D{T,2,1}, q::AbstractArray{T,4}; ptn::Union{AbstractArray{T,4},Nothing}=nothing) where T
    ptn === nothing && (ptn = ptnorm2_batch_2D(p; η=eps(T)))
    nx,ny,nc,nb = size(p)
    p = reshape(p, nx,ny,2,div(nc,2)*nb)
    q = reshape(q, nx,ny,2,div(nc,2)*nb)
    ptn = reshape(ptn, nx,ny,1,div(nc,2)*nb)
    q .= (T(1).-λ./ptn).*(ptn .>= λ).*p
    return reshape(q, nx,ny,nc,nb)
end

(::MixedNormBatch_2D{T,2,1})(p::AbstractArray{T,4}) where T = norm21_batch_2D(p)


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


# L2V norm

function L2V_norm_2D(; T::DataType=Float32, h::Tuple{S,S}=(T(1),T(1)), weight::Union{Nothing,AbstractLinearOperator}=nothing) where {S<:Number}
    ∇ = gradient_2D(; T=T, h=h)
    weight !== nothing ? (A∇ = weight*∇) : (A∇ = ∇)
    return MixedNorm_2D{T,2,2}()∘A∇
end


# LInfV norm

function LInfV_norm_2D(; T::DataType=Float32, h::Tuple{S,S}=(T(1),T(1)), weight::Union{Nothing,AbstractLinearOperator}=nothing) where {S<:Number}
    ∇ = gradient_2D(; T=T, h=h)
    weight !== nothing ? (A∇ = weight*∇) : (A∇ = ∇)
    return MixedNorm_2D{T,2,Inf}()∘A∇
end


# TV norm

function TV_norm_2D(; T::DataType=Float32, h::Tuple{S,S}=(T(1),T(1)), weight::Union{Nothing,AbstractLinearOperator}=nothing) where {S<:Number}
    ∇ = gradient_2D(; T=T, h=h)
    weight !== nothing ? (A∇ = weight*∇) : (A∇ = ∇)
    return MixedNorm_2D{T,2,1}()∘A∇
end

struct WeightedGradient_2D{T} <: AbstractLinearOperator{AbstractArray{T,2},AbstractArray{T,3}}
    P::ProjVectorField_2D{T}
    ∇::Gradient_2D{T}
end

AbstractLinearOperators.domain_size(A::WeightedGradient_2D) = AbstractLinearOperators.range_size(A.P)[1:2]
AbstractLinearOperators.range_size(A::WeightedGradient_2D) = AbstractLinearOperators.range_size(A.P)
AbstractLinearOperators.matvecprod(A::WeightedGradient_2D, u::AbstractArray{T,2}) where T = A.P*(A.∇*u)
AbstractLinearOperators.matvecprod_adj(A::WeightedGradient_2D, u::AbstractArray{T,3}) where T = adjoint(A.∇)*(adjoint(A.P)*u)

Base.:*(P::ProjVectorField_2D{T}, ∇::Gradient_2D{T}) where T = WeightedGradient_2D{T}(P, ∇)

Flux.gpu(A::WeightedGradient_2D{T}) where T = WeightedGradient_2D{T}(Flux.gpu(A.P), Flux.gpu(A.∇))
Flux.cpu(A::WeightedGradient_2D{T}) where T = WeightedGradient_2D{T}(Flux.cpu(A.P), Flux.cpu(A.∇))


# TV norm (batch)

function TV_norm_batch_2D(; T::DataType=Float32, h::Tuple{S,S}=(T(1),T(1)), weight::Union{Nothing,AbstractLinearOperator}=nothing) where {S<:Number}
    ∇ = gradient_batch_2D(; T=T, h=h)
    weight !== nothing ? (A∇ = weight*∇) : (A∇ = ∇)
    return MixedNormBatch_2D{T,2,1}()∘A∇
end


# Utils (2-D)

ptdot(v1::AbstractArray{T,3}, v2::AbstractArray{T,3}) where T = sum(conj.(v1).*v2; dims=3)[:,:,1]
ptnorm1(p::AbstractArray{T,3}; η::T=T(0)) where T = abs.(p[:,:,1])+abs.(p[:,:,2]).+abs(η)
ptnorm2(p::AbstractArray{T,3}; η::T=T(0)) where T = sqrt.(abs.(p[:,:,1]).^2+abs.(p[:,:,2]).^2 .+abs(η)^2)
ptnormInf(p::AbstractArray{T,3}; η::T=T(0)) where T = maximum(abs.(p).+abs(η); dims=3)[:,:,1]
function ptnorm2_batch(p::AbstractArray{T,4}; η::T=T(0)) where T
    nx,ny,nc,nb = size(p)
    p = reshape(p, nx,ny,2,div(nc,2)*nb)
    return reshape(sqrt.(abs.(p[:,:,1:1,:]).^2+abs.(p[:,:,2:2,:]).^2 .+abs(η)^2), nx,ny,div(nc,2),nb)
end

norm21(v::AbstractArray{T,3}; η::T=T(0)) where T = sum(ptnorm2(v; η=η))
norm22(v::AbstractArray{T,3}; η::T=T(0)) where T = sqrt(sum(ptnorm2(v; η=η).^2))
norm2Inf(v::AbstractArray{T,3}; η::T=T(0)) where T = maximum(ptnorm2(v; η=η))
function norm21_batch(v::AbstractArray{T,4}; η::T=T(0)) where T
    _,_,nc,nb = size(v)
    return reshape(sum(ptnorm2_batch(v; η=η); dims=(1,2)), div(nc,2), nb)
end


# Utils (3-D)

ptdot(v1::AbstractArray{T,4}, v2::AbstractArray{T,4}) where T = sum(conj.(v1).*v2; dims=4)[:,:,:,1]
ptnorm1(p::AbstractArray{T,4}; η::T=T(0)) where T = abs.(p[:,:,:,1])+abs.(p[:,:,:,2])+abs.(p[:,:,:,3]).+abs(η)
ptnorm2(p::AbstractArray{T,4}; η::T=T(0)) where T = sqrt.(abs.(p[:,:,:,1]).^2+abs.(p[:,:,:,2]).^2+abs.(p[:,:,:,3]).^2 .+abs(η)^2)
ptnormInf(p::AbstractArray{T,4}; η::T=T(0)) where T = maximum(abs.(p).+η; dims=4)[:,:,:,1]
function ptnorm2_batch(p::AbstractArray{T,5}; η::T=T(0)) where T
    nx,ny,nz,nc,nb = size(p)
    p = reshape(p, nx,ny,nz,2,div(nc,2)*nb)
    return reshape(sqrt.(abs.(p[:,:,:,1:1,:]).^2+abs.(p[:,:,:,2:2,:]).^2+abs.(p[:,:,:,3:3,:]).^2 .+abs(η)^2), nx,ny,nz,div(nc,2),nb)
end

norm21(v::AbstractArray{T,4}; η::T=T(0)) where T = sum(ptnorm2(v; η=η))
norm22(v::AbstractArray{T,4}; η::T=T(0)) where T = sqrt(sum(ptnorm2(v; η=η).^2))
norm2Inf(v::AbstractArray{T,4}; η::T=T(0)) where T = maximum(ptnorm2(v; η=η))
function norm21_batch(v::AbstractArray{T,5}; η::T=T(0)) where T
    _,_,_,nc,nb = size(v)
    return reshape(sum(ptnorm2_batch(v; η=η); dims=(1,2,3)), div(nc,2), nb)
end