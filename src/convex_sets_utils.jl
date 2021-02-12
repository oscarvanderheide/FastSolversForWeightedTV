#: Convex set utilities


export ℓball_2D, ell_ball, upperlim_constraints_2D, lowerlim_constraints_2D, box_constraints_2D, Aball_2D, tv_ball_2D, tsqv_ball_2D, bv_ball_2D


# ℓball_2D{p1,p2} ε-ball: C={x:||x||_{p1,p2}<=ε}

struct ℓball_2D{T,p1,p2}<:ConvexSet{T,3}
    ε::T
end

ell_ball(p1::Number, p2::Number, ε::T) where T = ℓball_2D{T,p1,p2}(ε)

project!(p::DT, C::ℓball_2D{T,2,1},   q::DT) where {T,DT<:AbstractArray{T,3}} = project!(p, C.ε, ell_norm(T,2,1),   q)
project!(p::DT, C::ℓball_2D{T,2,2},   q::DT) where {T,DT<:AbstractArray{T,3}} = project!(p, C.ε, ell_norm(T,2,2),   q)
project!(p::DT, C::ℓball_2D{T,2,Inf}, q::DT) where {T,DT<:AbstractArray{T,3}} = project!(p, C.ε, ell_norm(T,2,Inf), q)


# A-ball

struct Aball_2D{T,p1,p2}<:ConvexSet{T,2}
    A::AbstractLinearOperator
    ε::T
end

function project!(y::DT, C::Aball_2D{T,2,p2}, opt::OptimOptions{T}, x::DT; dual_est::Bool=false) where {T,p2,DT<:AbstractArray{T,2}}

    # Minimization function
    # f = leastsquares_misfit(adjoint(C.A), y)+C.ε*ell_norm(T,2,conjugate_exp(p2))
    f = leastsquares_misfit(adjoint(C.A), y)+conjugate(indicator(ell_ball(2,p2, C.ε)))

    # Minimization
    p = minimize(f, opt)

    # Dual to primal solution
    x .= y-adjoint(C.A)*p

    dual_est ? (return x, p) : (return x)

end

project(y::AbstractArray{T,2}, C::Aball_2D{T,2,p2}, opt::OptimOptions{T}; dual_est::Bool=false) where {T,p2} = project!(y, C, opt, similar(y); dual_est=dual_est)

# function conjugate_exp(p2::Number)::Number
#     p2 == 1   && (return Inf)
#     p2 == 2   && (return 2)
#     p2 == Inf && (return 1)
# end


# TV-related ball

tv_ball_2D(n::Tuple{Int64,Int64}, ε::T; h::Tuple{T,T}=(T(1),T(1)), gpu::Bool=false) where T = Aball_2D{T,2,1}(gradient_2D(n; h=h, gpu=gpu), ε)

tsqv_ball_2D(n::Tuple{Int64,Int64}, ε::T; h::Tuple{T,T}=(T(1),T(1)), gpu::Bool=false) where T = Aball_2D{T,2,2}(gradient_2D(n; h=h, gpu=gpu), ε)

bv_ball_2D(n::Tuple{Int64,Int64}, ε::T; h::Tuple{T,T}=(T(1),T(1)), gpu::Bool=false) where T = Aball_2D{T,2,Inf}(gradient_2D(n; h=h, gpu=gpu), ε)


# Structural TV-related ball

function tv_ball_2D(u::AbstractArray{T,2}, ε::T; h::Tuple{T,T}=(T(1),T(1)), η::T=T(0)) where T
    ∇ = gradient_2D(size(u); h=h, gpu=u isa CuArray)
    P = projvectorfield_2D(∇*u; η=η)
    return Aball_2D{T,2,1}(P*∇, ε)
end

function tsqv_ball_2D(u::AbstractArray{T,2}, ε::T; h::Tuple{T,T}=(T(1),T(1)), η::T=T(0)) where T
    ∇ = gradient_2D(size(u); h=h, gpu=u isa CuArray)
    P = projvectorfield_2D(∇*u; η=η)
    return Aball_2D{T,2,2}(P*∇, ε)
end

function bv_ball_2D(u::AbstractArray{T,2}, ε::T; h::Tuple{T,T}=(T(1),T(1)), η::T=T(0)) where T
    ∇ = gradient_2D(size(u); h=h, gpu=u isa CuArray)
    P = projvectorfield_2D(∇*u; η=η)
    return Aball_2D{T,2,Inf}(P*∇, ε)
end


# ElasticNetABall_2D ε-ball: C={x:||x||_{2,1}+μ^2/2*||x||_{2,2}^2<=ε}

struct ElasticNetABall_2D{T}<:ConvexSet{T,2}
    A::AbstractLinearOperator
    μ::T
    ε::T
end

function project!(y::DT, C::ElasticNetABall_2D{T}, opt::OptimOptions{T}, x::DT; dual_est::Bool=false) where {T,DT<:AbstractArray{T,2}}

    # Minimization function
    f = leastsquares_misfit(adjoint(C.A), y)+conjugate(indicator(elastic_net_vect(C.μ), C.ε))

    # Minimization
    p = minimize(f, opt)

    # Dual to primal solution
    x .= y-adjoint(C.A)*p

    dual_est ? (return x, p) : (return x)

end

### Utils for root-finding elastic net

function pareto_search_elnet(ptn::AbstractArray{T,2}) where T
    obj_fun = δ->objfun_paretosearch_proj21(δ, ptn)
    return find_zero(obj_fun, (T(0), maximum(ptn)))
end

function objfun_paretosearch_elnet(δ::T, ptn::Array{T,2}) where T
    ptn_ = ptn[ptn.>=δ]
    return T(1)-sum(ptn_)+length(ptn_)*δ
end

function objfun_paretosearch_elnet(δ::T, ptn::CuArray{T,2}) where T
    ptn_ = ptn[ptn.>=δ]
    return T(1)-sum(ptn_)+length(ptn_)*δ
end


# Box constraints

abstract type AbstractValueConstraints2D{T}<:ConvexSet{T,2} end

struct LowerLimConstraints2D{T}<:AbstractValueConstraints2D{T}
    a::T
end

function project!(x::Array{T,2}, C::LowerLimConstraints2D{T}, y::Array{T,2}) where T
    @inbounds for i = 1:length(x)
        x[i] < C.a ? (y[i] = C.a) : (y[i] = x[i])
    end
    return y
end

function project!(x::CuArray{T,2}, C::LowerLimConstraints2D{T}, y::CuArray{T,2}) where T
    idx = x.<C.a
    y[idx] .= C.a
    y[(!).(idx)] .= x[(!).(idx)]
    return y
end

struct UpperLimConstraints2D{T}<:AbstractValueConstraints2D{T}
    b::T
end

function project!(x::Array{T,2}, C::UpperLimConstraints2D{T}, y::Array{T,2}) where T
    @inbounds for i = 1:length(x)
        x[i] > C.b ? (y[i] = C.b) : (y[i] = x[i])
    end
    return y
end

function project!(x::CuArray{T,2}, C::UpperLimConstraints2D{T}, y::CuArray{T,2}) where T
    idx = x.>C.b
    y[idx] .= C.b
    y[(!).(idx)] .= x[(!).(idx)]
    return y
end

struct BoxConstraints2D{T}<:AbstractValueConstraints2D{T}
    a::T
    b::T
end

function project!(x::Array{T,2}, C::BoxConstraints2D{T}, y::Array{T,2}) where T
    @inbounds for i = 1:length(x)
        y[i] = x[i]
        x[i] > C.b && (y[i] = C.b)
        x[i] < C.a && (y[i] = C.a)
    end
    return y
end

function project!(x::CuArray{T,2}, C::BoxConstraints2D{T}, y::CuArray{T,2}) where T
    idx_a = x.<C.a
    idx_b = x.>C.b
    y[idx_a] .= C.a
    y[idx_b] .= C.b
    y[(!).(idx_a) || (!).(idx_b)] .= x[(!).(idx_a) || (!).(idx_b)]
    return y
end

upperlim_constraints_2D(b::T) where T = UpperLimConstraints2D{T}(b)
lowerlim_constraints_2D(a::T) where T = LowerLimConstraints2D{T}(a)
box_constraints_2D(a::T, b::T) where T = BoxConstraints2D{T}(a, b)