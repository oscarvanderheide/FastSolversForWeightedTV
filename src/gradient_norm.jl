#: Utilities for norm functions

export gradient_norm, gradient_norm_batch


# Gradient mixed norm

function gradient_norm(N1::Number, N2::Number, n::NTuple{D,Int64}, h::NTuple{D,T}, opt::AbstractOptimizer; weight::Union{Nothing,AbstractLinearOperator}=nothing, pareto_tol::Union{Nothing,Real}=nothing, complex::Bool=false) where {D,T<:Real}
    ∇ = gradient_operator(n, h; complex=complex)
    weight !== nothing ? (A∇ = weight*∇) : (A∇ = ∇)
    complex ? (CT = Complex{T}) : (CT = T)
    return weighted_prox(mixed_norm(CT, D, N1, N2; pareto_tol=pareto_tol), A∇, opt)
end


# Gradient mixed norm (2-D, batch)

function gradient_norm_batch(N1::Number, N2::Number, n::NTuple{D,Int64}, nc::Int64, nb::Int64, h::NTuple{2,T}, opt::AbstractOptimizer; weight::Union{Nothing,AbstractLinearOperator}=nothing, pareto_tol::Union{Nothing,Real}=nothing, complex::Bool=false) where {T<:Real,D}
    ∇ = gradient_operator_batch(n, nc, nb, h; complex=complex)
    weight !== nothing ? (A∇ = weight*∇) : (A∇ = ∇)
    complex ? (CT = Complex{T}) : (CT = T)
    return weighted_prox(mixed_norm_batch(CT, D, N1, N2; pareto_tol=pareto_tol), A∇, opt)
end