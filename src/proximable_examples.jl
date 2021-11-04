#: Utilities for norm functions

export ptdot, ptnorm1, ptnorm2, ptnormInf
export norm_2D, norm_batch_2D, L2V_norm_2D, LInfV_norm_2D, TV_norm_2D, TV_norm_batch_2D
export norm_3D,                L2V_norm_3D, LInfV_norm_3D, TV_norm_3D


# Mixed norm (2-D)

struct MixedNorm_2D{T,N1,N2}<:ProximableFunction{T,3} end

struct WeightedMixedNorm_2D{T,N1,N2,WT<:AbstractLinearOperator}<:ProximableFunction{T,3}
    weight::WT
    solver_opt::OptimOptions
end

norm_2D(N1::Number, N2::Number; T::DataType=Float32) = MixedNorm_2D{T,N1,N2}()


# Mixed norm (3-D)

struct MixedNorm_3D{T,N1,N2}<:ProximableFunction{T,4} end

struct WeightedMixedNorm_3D{T,N1,N2,WT<:AbstractLinearOperator}<:ProximableFunction{T,4}
    weight::WT
    solver_opt::OptimOptions
end

norm_3D(N1::Number, N2::Number; T::DataType=Float32) = MixedNorm_3D{T,N1,N2}()


# Mixed norm (2-D, batch)

struct MixedNormBatch_2D{T,N1,N2}<:ProximableFunction{T,4} end

norm_batch_2D(N1::Number, N2::Number; T::DataType=Float32) = MixedNormBatch_2D{T,N1,N2}()


# L22 norm (2-D)

function proxy!(p::AbstractArray{T,3}, λ::T, ::MixedNorm_2D{T,2,2}, q::AbstractArray{T,3}) where T
    np = norm22(p)
    np <= λ ? (return q .= T(0)) : (return q .= (T(1)-λ/np)*p)
end

function project!(p::AbstractArray{T,3}, ε::T, ::MixedNorm_2D{T,2,2}, q::AbstractArray{T,3}) where T
    np = norm22(p)
    np <= ε ? (return q .= p) : (return q .= ε*p/np)
end

(::MixedNorm_2D{T,2,2})(p::AbstractArray{T,3}) where T = norm(p)


# L22 norm (3-D)

function proxy!(p::AbstractArray{T,4}, λ::T, ::MixedNorm_3D{T,2,2}, q::AbstractArray{T,4}) where T
    np = norm22(p)
    np <= λ ? (return q .= T(0)) : (return q .= (T(1)-λ/np)*p)
end

function project!(p::AbstractArray{T,4}, ε::T, ::MixedNorm_3D{T,2,2}, q::AbstractArray{T,4}) where T
    np = norm22(p)
    np <= ε ? (return q .= p) : (return q .= ε*p/np)
end

(::MixedNorm_3D{T,2,2})(p::AbstractArray{T,4}) where T = norm(p)


# L21 norm (2-D)

function proxy!(p::AbstractArray{T,3}, λ::T, ::MixedNorm_2D{T,2,1}, q::AbstractArray{T,3}; ptn::Union{AbstractArray{T,2},Nothing}=nothing) where T
    ptn === nothing && (ptn = ptnorm2(p; η=eps(T)))
    return q .= (T(1).-λ./ptn).*(ptn .>= λ).*p
end

function project!(p::AbstractArray{T,3}, ε::T, g::MixedNorm_2D{T,2,1}, q::AbstractArray{T,3}) where T
    ptn = ptnorm2(p; η=eps(T))
    sum(ptn) <= ε && (return q .= p)
    λ = pareto_search_proj21(ptn, ε)
    return proxy!(p, λ, g, q; ptn=ptn)
end

(::MixedNorm_2D{T,2,1})(p::AbstractArray{T,3}) where T = norm21(p)


# Pareto-search routines

# pareto_search_proj21(ptn::AbstractArray{T,N}, ε::T) where {T,N} = T(solve(ZeroProblem(λ -> obj_pareto_search_proj21(λ, ptn, ε), (T(0), maximum(ptn))), Roots.Brent()))
pareto_search_proj21(ptn::AbstractArray{T,N}, ε::T) where {T,N} = find_root(ptn, ε; xrtol=T(1e-3))

# obj_pareto_search_proj21(λ::T, ptn::AbstractArray{T,N}, ε::T) where {T,N} = sum(Flux.relu.(ptn.-λ))-ε

function find_root(ptn::AbstractArray{T,N}, ε::T; maxiter::Int64=length(ptn), atol::T=eps(T), xrtol::T=eps(T)) where {T,N}
    function f(λ::T; grad::Bool=false) where T
        x = ptn.-λ
        fval = sum(Flux.relu.(x))-ε
        ~grad ? (return fval) : (return fval, -T(sum(∇relu(x))))
    end
    a = T(0)
    for i = 1:maxiter
        fa, dfa = f(a; grad=true)
        a1 = a-fa/dfa
        ((abs(fa) <= atol) || (abs(a1-a)/abs(a) <= xrtol)) && break
        a = a1
    end
    return a
end
∇relu(x::AbstractArray{T,N}) where {T,N} = (x .>= T(0))


# L21 norm (3-D)

function proxy!(p::AbstractArray{T,4}, λ::T, ::MixedNorm_3D{T,2,1}, q::AbstractArray{T,4}; ptn::Union{AbstractArray{T,3},Nothing}=nothing) where T
    ptn === nothing && (ptn = ptnorm2(p; η=eps(T)))
    return q .= (T(1).-λ./ptn).*(ptn .>= λ).*p
end

function project!(p::AbstractArray{T,4}, ε::T, g::MixedNorm_3D{T,2,1}, q::AbstractArray{T,4}) where T
    ptn = ptnorm2(p; η=eps(T))
    sum(ptn) <= ε && (return q .= p)
    λ = pareto_search_proj21(ptn, ε)
    return proxy!(p, λ, g, q; ptn=ptn)
end

(::MixedNorm_3D{T,2,1})(p::AbstractArray{T,4}) where T = norm21(p)


# L21 norm (2-D, batch)

function proxy!(p::AbstractArray{T,4}, λ::T, ::MixedNormBatch_2D{T,2,1}, q::AbstractArray{T,4}; ptn::Union{AbstractArray{T,4},Nothing}=nothing) where T
    ptn === nothing && (ptn = ptnorm2_batch(p; η=eps(T)))
    nx,ny,nc,nb = size(p)
    p = reshape(p, nx,ny,2,div(nc,2)*nb)
    q = reshape(q, nx,ny,2,div(nc,2)*nb)
    ptn = reshape(ptn, nx,ny,1,div(nc,2)*nb)
    q .= (T(1).-λ./ptn).*(ptn .>= λ).*p
    return reshape(q, nx,ny,nc,nb)
end

(::MixedNormBatch_2D{T,2,1})(p::AbstractArray{T,4}) where T = norm21_batch(p)


# L2Inf norm (2-D)

function proxy!(p::AbstractArray{T,3}, λ::T, ::MixedNorm_2D{T,2,Inf}, q::AbstractArray{T,3}) where T
    project!(p, λ, norm_2D(2,1; T=T), q)
    return q .= p.-q
end

function project!(p::AbstractArray{T,3}, ε::T, ::MixedNorm_2D{T,2,Inf}, q::AbstractArray{T,3}) where T
    ptn = ptnorm2(p; η=eps(T))
    val = ptn .>= ε
    q .= p.*(ε*val./ptn+(!).(val))
    return q
end

(::MixedNorm_2D{T,2,Inf})(p::AbstractArray{T,3}) where T = norm2Inf(p)


# L2Inf norm (3-D)

function proxy!(p::AbstractArray{T,4}, λ::T, ::MixedNorm_3D{T,2,Inf}, q::AbstractArray{T,4}) where T
    project!(p, λ, norm_3D(2,1; T=T), q)
    return q .= p.-q
end

function project!(p::AbstractArray{T,4}, ε::T, ::MixedNorm_3D{T,2,Inf}, q::AbstractArray{T,4}) where T
    ptn = ptnorm2(p; η=eps(T))
    val = ptn .>= ε
    q .= p.*(ε*val./ptn+(!).(val))
    return q
end

(::MixedNorm_3D{T,2,Inf})(p::AbstractArray{T,4}) where T = norm2Inf(p)


# L2V norm

function L2V_norm_2D(n::NTuple{2,Int64}; T::DataType=Float32, h::Tuple{S,S}=(T(1),T(1)), weight::Union{Nothing,AbstractLinearOperator}=nothing) where {S<:Number}
    ∇ = gradient_2D(n; T=T, h=h)
    weight !== nothing ? (A∇ = weight*∇) : (A∇ = ∇)
    return MixedNorm_2D{T,2,2}()∘A∇
end


# L2V norm (3-D)

function L2V_norm_3D(n::NTuple{3,Int64}; T::DataType=Float32, h::Tuple{S,S,S}=(T(1),T(1),T(1)), weight::Union{Nothing,AbstractLinearOperator}=nothing) where {S<:Number}
    ∇ = gradient_3D(n; T=T, h=h)
    weight !== nothing ? (A∇ = weight*∇) : (A∇ = ∇)
    return MixedNorm_3D{T,2,2}()∘A∇
end


# LInfV norm

function LInfV_norm_2D(n::NTuple{2,Int64}; T::DataType=Float32, h::Tuple{S,S}=(T(1),T(1)), weight::Union{Nothing,AbstractLinearOperator}=nothing) where {S<:Number}
    ∇ = gradient_2D(n; T=T, h=h)
    weight !== nothing ? (A∇ = weight*∇) : (A∇ = ∇)
    return MixedNorm_2D{T,2,Inf}()∘A∇
end


# LInfV norm (3-D)

function LInfV_norm_3D(n::NTuple{3,Int64}; T::DataType=Float32, h::Tuple{S,S,S}=(T(1),T(1),T(1)), weight::Union{Nothing,AbstractLinearOperator}=nothing) where {S<:Number}
    ∇ = gradient_3D(n; T=T, h=h)
    weight !== nothing ? (A∇ = weight*∇) : (A∇ = ∇)
    return MixedNorm_3D{T,2,Inf}()∘A∇
end


# TV norm (2-D)

function TV_norm_2D(n::NTuple{2,Int64}; T::DataType=Float32, h::Tuple{S,S}=(T(1),T(1)), weight::Union{Nothing,AbstractLinearOperator}=nothing) where {S<:Number}
    ∇ = gradient_2D(n; T=T, h=h)
    weight !== nothing ? (A∇ = weight*∇) : (A∇ = ∇)
    return MixedNorm_2D{T,2,1}()∘A∇
end


# TV norm (3-D)

function TV_norm_3D(n::NTuple{3,Int64}; T::DataType=Float32, h::Tuple{S,S,S}=(T(1),T(1),T(1)), weight::Union{Nothing,AbstractLinearOperator}=nothing) where {S<:Number}
    ∇ = gradient_3D(n; T=T, h=h)
    weight !== nothing ? (A∇ = weight*∇) : (A∇ = ∇)
    return MixedNorm_3D{T,2,1}()∘A∇
end


# Weighted gradient (2-D)

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


# Weighted gradient (3-D)

struct WeightedGradient_3D{T} <: AbstractLinearOperator{AbstractArray{T,3},AbstractArray{T,4}}
    P::ProjVectorField_3D{T}
    ∇::Gradient_3D{T}
end

AbstractLinearOperators.domain_size(A::WeightedGradient_3D) = AbstractLinearOperators.range_size(A.P)[1:2]
AbstractLinearOperators.range_size(A::WeightedGradient_3D) = AbstractLinearOperators.range_size(A.P)
AbstractLinearOperators.matvecprod(A::WeightedGradient_3D, u::AbstractArray{T,3}) where T = A.P*(A.∇*u)
AbstractLinearOperators.matvecprod_adj(A::WeightedGradient_3D, u::AbstractArray{T,4}) where T = adjoint(A.∇)*(adjoint(A.P)*u)

Base.:*(P::ProjVectorField_3D{T}, ∇::Gradient_3D{T}) where T = WeightedGradient_3D{T}(P, ∇)

Flux.gpu(A::WeightedGradient_3D{T}) where T = WeightedGradient_3D{T}(Flux.gpu(A.P), Flux.gpu(A.∇))
Flux.cpu(A::WeightedGradient_3D{T}) where T = WeightedGradient_3D{T}(Flux.cpu(A.P), Flux.cpu(A.∇))


# TV norm (2-D, batch)

function TV_norm_batch_2D(n::NTuple{4,Int64}; T::DataType=Float32, h::Tuple{S,S}=(T(1),T(1)), weight::Union{Nothing,AbstractLinearOperator}=nothing) where {S<:Number}
    ∇ = gradient_batch_2D(n; T=T, h=h)
    weight !== nothing ? (A∇ = weight*∇) : (A∇ = ∇)
    return MixedNormBatch_2D{T,2,1}()∘A∇
end


# Utils

ptdot(v1::AbstractArray{T,N}, v2::AbstractArray{T,N}) where {T,N} = dropdims(sum(conj.(v1).*v2; dims=N); dims=N)
ptnorm1(p::AbstractArray{T,N}; η::T=T(0)) where {T,N} = dropdims(sum(abs.(p).+abs(η); dims=N); dims=N)
ptnorm2(p::AbstractArray{T,N}; η::T=T(0)) where {T,N} = dropdims(sqrt.(sum(abs.(p).^2 .+abs(η)^2; dims=N)); dims=N)
ptnormInf(p::AbstractArray{T,N}; η::T=T(0)) where {T,N} = dropdims(sqrt.(maximum(abs.(p).+abs(η); dims=N)); dims=N)

function ptnorm2_batch(p::AbstractArray{T,4}; η::T=T(0)) where T
    nx,ny,nc,nb = size(p)
    p = reshape(p, nx,ny,2,div(nc,2)*nb)
    return reshape(sqrt.(abs.(p[:,:,1:1,:]).^2+abs.(p[:,:,2:2,:]).^2 .+abs(η)^2), nx,ny,div(nc,2),nb)
end
function ptnorm2_batch(p::AbstractArray{T,5}; η::T=T(0)) where T
    nx,ny,nz,nc,nb = size(p)
    p = reshape(p, nx,ny,nz,2,div(nc,2)*nb)
    return reshape(sqrt.(abs.(p[:,:,:,1:1,:]).^2+abs.(p[:,:,:,2:2,:]).^2+abs.(p[:,:,:,3:3,:]).^2 .+abs(η)^2), nx,ny,nz,div(nc,2),nb)
end

norm21(v::AbstractArray{T,N}; η::T=T(0)) where {T,N} = sum(ptnorm2(v; η=η))
norm22(v::AbstractArray{T,N}; η::T=T(0)) where {T,N} = sqrt(sum(ptnorm2(v; η=η).^2))
norm2Inf(v::AbstractArray{T,N}; η::T=T(0)) where {T,N} = maximum(ptnorm2(v; η=η))
function norm21_batch(v::AbstractArray{T,4}; η::T=T(0)) where T
    _,_,nc,nb = size(v)
    return reshape(sum(ptnorm2_batch(v; η=η); dims=(1,2)), div(nc,2), nb)
end
function norm21_batch(v::AbstractArray{T,5}; η::T=T(0)) where T
    _,_,_,nc,nb = size(v)
    return reshape(sum(ptnorm2_batch(v; η=η); dims=(1,2,3)), div(nc,2), nb)
end