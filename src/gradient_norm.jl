#: Utilities for norm functions

export gradient_norm


# Gradient mixed norm

function gradient_norm(P1::Number, P2::Number, n::NTuple{N,Int64}, h::NTuple{N,T}; weight::Union{Nothing,AbstractLinearOperator}=nothing, pareto_tol::Union{Nothing,Real}=nothing, complex::Bool=false, gpu::Bool=false) where {T<:Real,N}
    ∇ = gradient_operator(n, h; complex=complex, gpu=gpu)
    weight !== nothing ? (A∇ = weight*∇) : (A∇ = ∇)
    complex ? (CT = Complex{T}) : (CT = T)
    return weighted_prox(mixed_norm(CT,N,P1,P2; pareto_tol=pareto_tol), A∇)
end