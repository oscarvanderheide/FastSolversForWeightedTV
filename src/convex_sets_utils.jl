#: Convex set utilities


export ℓball_2D, ell_ball, upperlim_constraints_2D, lowerlim_constraints_2D, box_constraints_2D, Aball_2D, tv_ball_2D, tsqv_ball_2D, bv_ball_2D, elnettv_ball_2D, project_debug


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
    opt::OptimOptions
end

function project!(y::DT, C::Aball_2D{T,2,p2}, x::DT; update_dual_estimate::Bool=false) where {T,p2,DT<:AbstractArray{T,2}}

    # Minimization function
    f = leastsquares_misfit(adjoint(C.A), y)+conjugate(indicator(ell_norm(T, 2, p2), C.ε))

    # Minimization (dual variable)
    y isa CuArray ? (p0 = CUDA.zeros(T, size(y)..., 2)) : (p0 = zeros(T, size(y)..., 2))
    p = minimize(f, p0, C.opt); update_dual_estimate && (C.opt.initial_estimate .= p)

    # Dual to primal solution
    x .= y-adjoint(C.A)*p

    return x

end

function project_debug!(y::DT, C::Aball_2D{T,2,p2}, x::DT; update_dual_estimate::Bool=false) where {T,p2,DT<:AbstractArray{T,2}}

    # Minimization function
    f = leastsquares_misfit(adjoint(C.A), y)+conjugate(indicator(ell_norm(T, 2, p2), C.ε))

    # Minimization (dual variable)
    p, fval_hist, err_rel = minimize_debug(f, C.opt); update_dual_estimate && (C.opt.initial_estimate .= p)

    # Dual to primal solution
    x .= y-adjoint(C.A)*p

    return x, fval_hist, err_rel

end

project(y::AbstractArray{T,2}, C::Aball_2D{T,2,p2}; update_dual_estimate::Bool=false) where {T,p2} = project!(y, C, similar(y); update_dual_estimate=update_dual_estimate)

project_debug(y::AbstractArray{T,2}, C::Aball_2D{T,2,p2}; update_dual_estimate::Bool=false) where {T,p2} = project_debug!(y, C, similar(y); update_dual_estimate=update_dual_estimate)


# TV-related ball

function tv_ball_2D(n::Tuple{Int64,Int64}, ε::T; h::Tuple{T,T}=(T(1),T(1)), gpu::Bool=false, opt::Union{Nothing,OptimOptions}=nothing) where T
    if opt === nothing
        gpu ? (p0 = CUDA.zeros(T, n..., 3)) : (p0 = zeros(T, n..., 3))
        opt = opt_fista(; initial_estimate=p0, steplength=1f0/8f0, niter=10)
    end
    return Aball_2D{T,2,1}(gradient_2D(n; h=h, gpu=gpu), ε, opt)
end

function tsqv_ball_2D(n::Tuple{Int64,Int64}, ε::T; h::Tuple{T,T}=(T(1),T(1)), gpu::Bool=false, opt::Union{Nothing,OptimOptions}=nothing) where T
    if opt === nothing
        gpu ? (p0 = CUDA.zeros(T, n..., 3)) : (p0 = zeros(T, n..., 3))
        opt = opt_fista(; initial_estimate=p0, steplength=1f0/8f0, niter=10)
    end
    return Aball_2D{T,2,2}(gradient_2D(n; h=h, gpu=gpu), ε, opt)
end

function bv_ball_2D(n::Tuple{Int64,Int64}, ε::T; h::Tuple{T,T}=(T(1),T(1)), gpu::Bool=false, opt::Union{Nothing,OptimOptions}=nothing) where T
    if opt === nothing
        gpu ? (p0 = CUDA.zeros(T, n..., 3)) : (p0 = zeros(T, n..., 3))
        opt = opt_fista(; initial_estimate=p0, steplength=1f0/8f0, niter=10)
    end
    return Aball_2D{T,2,Inf}(gradient_2D(n; h=h, gpu=gpu), ε, opt)
end


# Structural TV-related ball

function tv_ball_2D(u::AbstractArray{T,2}, η::T, ε::T; h::Tuple{T,T}=(T(1),T(1)), opt::Union{Nothing,OptimOptions}=nothing) where T
    opt === nothing && (opt = opt_fista(; initial_estimate=similar(u, size(u)..., 3), steplength=1f0/8f0, niter=10))
    ∇ = gradient_2D(size(u); h=h, gpu=u isa CuArray)
    P = projvectorfield_2D(∇*u; η=η)
    return Aball_2D{T,2,1}(P*∇, ε, opt)
end

function tsqv_ball_2D(u::AbstractArray{T,2}, η::T, ε::T; h::Tuple{T,T}=(T(1),T(1)), opt::Union{Nothing,OptimOptions}=nothing) where T
    opt === nothing && (opt = opt_fista(; initial_estimate=similar(u, size(u)..., 3), steplength=1f0/8f0, niter=10))
    ∇ = gradient_2D(size(u); h=h, gpu=u isa CuArray)
    P = projvectorfield_2D(∇*u; η=η)
    return Aball_2D{T,2,2}(P*∇, ε, opt)
end

function bv_ball_2D(u::AbstractArray{T,2}, η::T, ε::T; h::Tuple{T,T}=(T(1),T(1)), opt::Union{Nothing,OptimOptions}=nothing) where T
    opt === nothing && (opt = opt_fista(; initial_estimate=similar(u, size(u)..., 3), steplength=1f0/8f0, niter=10))
    ∇ = gradient_2D(size(u); h=h, gpu=u isa CuArray)
    P = projvectorfield_2D(∇*u; η=η)
    return Aball_2D{T,2,Inf}(P*∇, ε, opt)
end


# ElasticNetABall_2D ε-ball: C={x:||x||_{2,1}+μ^2/2*||x||_{2,2}^2<=ε}

struct ElasticNetABall_2D{T}<:ConvexSet{T,2}
    A::AbstractLinearOperator
    μ::T
    ε::T
    opt::OptimOptions
end

function project!(y::DT, C::ElasticNetABall_2D{T}, x::DT; update_dual_estimate::Bool=false) where {T,DT<:AbstractArray{T,2}}

    # Minimization function
    f = leastsquares_misfit(adjoint(C.A), y)+conjugate(indicator(elastic_net_vect(C.μ), C.ε))

    # Minimization
    p = minimize(f, C.opt); update_dual_estimate && (C.opt.initial_estimate .= p)

    # Dual to primal solution
    x .= y-adjoint(C.A)*p
    return x

end

function project_debug!(y::DT, C::ElasticNetABall_2D{T}, x::DT; update_dual_estimate::Bool=false) where {T,DT<:AbstractArray{T,2}}

    # Minimization function
    f = leastsquares_misfit(adjoint(C.A), y)+conjugate(indicator(elastic_net_vect(C.μ), C.ε))

    # Minimization
    p, fval_hist, err_rel = minimize_debug(f, C.opt); update_dual_estimate && (C.opt.initial_estimate .= p)

    # Dual to primal solution
    x .= y-adjoint(C.A)*p
    return x, fval_hist, err_rel

end

project(y::AbstractArray{T,2}, C::ElasticNetABall_2D{T}; update_dual_estimate::Bool=false) where T = project!(y, C, similar(y); update_dual_estimate=update_dual_estimate)

project_debug(y::AbstractArray{T,2}, C::ElasticNetABall_2D{T}; update_dual_estimate::Bool=false) where T = project_debug!(y, C, similar(y); update_dual_estimate=update_dual_estimate)

function elnettv_ball_2D(n::Tuple{Int64,Int64}, μ::T, ε::T; h::Tuple{T,T}=(T(1),T(1)), opt::Union{Nothing,OptimOptions}=nothing, gpu::Bool=false) where T
    if opt === nothing
        gpu ? (p0 = CUDA.zeros(T, n..., 3)) : (p0 = zeros(T, n..., 3))
        opt =  opt_fista(; initial_estimate=p0, steplength=1f0/8f0, niter=10)
    end
    return ElasticNetABall_2D{T}(gradient_2D(n; h=h, gpu=gpu), μ, ε, opt)
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