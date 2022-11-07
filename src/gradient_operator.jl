#: Gradient operator via convolution

export gradient_operator, gradient_eval


# Generic convolutional operator

struct ConvolutionalOperator{T,N}<:AbstractLinearOperator{T,N,N}
    cdims::DenseConvDims
    stencil::AbstractArray{T,N}
end

AbstractLinearOperators.domain_size(A::ConvolutionalOperator) = NNlib.input_size(A.cdims)
AbstractLinearOperators.range_size(A::ConvolutionalOperator) = NNlib.output_size(A.cdims)
AbstractLinearOperators.matvecprod(A::ConvolutionalOperator{T,N}, u::AbstractArray{T,N}) where {T,N} = conv(u, A.stencil)
AbstractLinearOperators.matvecprod(A::ConvolutionalOperator{T,N}, u::CuArray{T,N}) where {T<:Complex,N} = conv(real(u), real(A.stencil))+im*conv(imag(u), real(A.stencil)) ### workaround for complex convolution for cuarrays
AbstractLinearOperators.matvecprod_adj(A::ConvolutionalOperator{T,N}, v::AbstractArray{T,N}) where {T,N} = ∇conv_data(v, A.stencil, A.cdims)
AbstractLinearOperators.matvecprod_adj(A::ConvolutionalOperator{T,N}, v::CuArray{T,N}) where {T<:Complex,N} = ∇conv_data(real(v), real(A.stencil), A.cdims)+im*∇conv_data(imag(v), real(A.stencil), A.cdims) ### workaround for complex convolution for cuarrays


# Gradient operator

function gradient_operator(n::NTuple{dim,Int64}, h::NTuple{dim,T}; complex::Bool=true, gpu::Bool=false) where {dim,T<:Real}

    # Gradient operator (w/out shaping)
    stencil = gradient_stencil(h; complex=complex, gpu=gpu)
    cdims = DenseConvDims((n..., 1, 1), size(stencil))
    complex ? (CT = Complex{T}) : (CT = T)
    ∇_ = ConvolutionalOperator{CT,dim+2}(cdims, stencil)

    # Reshaping operator
    Rin  = reshape_operator(CT, n, (n...,1,1))
    Rout = reshape_operator(CT, ((n.-1)...,dim,1), ((n.-1)...,dim))

    return Rout*∇_*Rin

end


# Gradient stencil utils

function gradient_stencil(h::NTuple{D,T}; complex::Bool=false, gpu::Bool=false) where {D,T<:Real}
    k = tuple(2*ones(Int64, D)...)
    complex ? (CT = Complex{T}) : (CT = T)
    stencil = zeros(CT,k...,1,D)
    for i = 1:D
        idx = Vector{UnitRange{Int64}}(undef,D); fill!(idx, 2:2)
        idx[i] = 1:2
        view(stencil, tuple(idx...)...,1,i)[1:2] .= [T(1)/h[i]; T(-1)/h[i]]
    end
    gpu ? (return convert(CuArray, stencil)) : (return stencil)
end


# Stencil-free gradient evaluation

function gradient_eval(u::AbstractArray{CT,D}, h::NTuple{D,T}) where {T<:Real,CT<:RealOrComplex{T},D}
    n = size(u)
    Du = similar(u, (n.-1)..., D)
    idx = Vector{Any}(undef,D)
    @inbounds for d = 1:D
        idx[d] = 1:n[d]-1
    end
    idx_p1 = Vector{Any}(undef,D)
    @inbounds for d = 1:D
        @inbounds for d_ = 1:D
            (d_ != d) ? (idx_p1[d_] = 1:n[d_]-1) : (idx_p1[d_] = 2:n[d_])
        end
        Du[idx..., d] .= (u[idx_p1...]-u[idx...])/h[d]
    end
    return Du
end