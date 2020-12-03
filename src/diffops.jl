# Differential operator functions 


using LinearMaps, FFTW, NNlib, CUDA, Flux
export Fourier_op, ConvLinearMap, derivative_ops, derivative_op, laplacian_op, gradient_op, divergence_op


## Wrappers for abstract matrices

Fourier_op(n::Tuple{Int64,Int64}) = LinearMap{T}(u -> fft(u), u -> ifft(u), prod(n), prod(n))

function derivative_ops(n::Tuple{Int64,Int64}; h::Tuple{T,T}=(T(1),T(1)), flag_gpu::Bool=false)
    Dx = derivative_op(n, "x"; h=h[1], flag_gpu=flag_gpu)
    Dy = derivative_op(n, "y"; h=h[2], flag_gpu=flag_gpu)
    return Dx, Dy
end

function derivative_op(n::Tuple{Int64,Int64}, varname::String; h::T=T(1), flag_gpu::Bool=false)
    if varname == "x"
        Dx = reshape([T(-1)/h T(1)/h], 2, 1); flag_gpu && (Dx = Dx |> gpu)
        return ConvLinearMap(n, Dx; pad0_output=Pad(0, 1, 0, 0))
    end
    if varname == "y"
        Dy = reshape([T(-1)/h T(1)/h], 1, 2); flag_gpu && (Dy = Dy |> gpu)
        return ConvLinearMap(n, Dy; pad0_output=Pad(0, 0, 0, 1))
    end
    throw(ArgumentError("Derivative variables must be either x or y"))
end

function laplacian_op(n::Tuple{Int64,Int64}; h::Tuple{T,T}=(T(1),T(1)), impl::String="mimetic", flag_gpu::Bool=false)
    if impl == "mimetic"
        return laplacian_mimetic_op(n; h=h, flag_gpu=flag_gpu)
    elseif impl == "fast"
        return laplacian_fast_op(n; h=h, flag_gpu=flag_gpu)
    else
        throw(ArgumentError("Implementation requested not available"))
    end
end

function laplacian_mimetic_op(n::Tuple{Int64,Int64}; h::Tuple{T,T}=(T(1),T(1)), flag_gpu::Bool=false)
    Dx, Dy = derivative_ops(n; h=h, flag_gpu=flag_gpu)
    Dx_T = transpose(Dx); Dy_T = transpose(Dy);
    Δ(u) = -Dx_T*(Dx*u)-Dy_T*(Dy*u)
    return LinearMap{T}(Δ, Δ, prod(n), prod(n))
end

function laplacian_fast_op(n::Tuple{Int64,Int64}; h::Tuple{T,T}=(T(1),T(1)), flag_gpu::Bool=false)
    Δ = [T(0) T(1)/h[1]^2 T(0); T(1)/h[2]^2 -T(2)*(T(1)/h[1]^2+T(1)/h[2]^2) T(1)/h[2]^2; T(0) T(1)/h[1]^2 T(0)]; flag_gpu && (Δ = Δ |> gpu)
    return ConvLinearMap(n, Δ; pad0_output=Pad(1, 1, 1, 1))
end

function gradient_op(n::Tuple{Int64,Int64}; h::Tuple{T,T}=(T(1),T(1)), flag_gpu::Bool=false)
    Dx, Dy = derivative_ops(n; h=h, flag_gpu=flag_gpu)
    Dx_T = transpose(Dx); Dy_T = transpose(Dy);
    return LinearMap{T}(u -> VectorField2D(Dx*u, Dy*u), v -> Dx_T*v.x+Dy_T*v.y, 2*prod(n), prod(n))
end

function divergence_op(n::Tuple{Int64,Int64}; h::Tuple{T,T}=(T(1),T(1)), flag_gpu::Bool=false)
    return transpose(-gradient_op(n; h=h, flag_gpu=flag_gpu))
end

function *(A::LinearMap{T}, u::Union{AbstractArray{T,2}, AbstractArray{CT,2}, VectorField2D})
    return A.f(u)
end

function *(A::LinearMaps.TransposeMap{T}, u::Union{AbstractArray{T,2}, AbstractArray{CT,2}, VectorField2D})
    return A.lmap.fc(u)
end

function *(A::LinearMaps.ScaledMap{T}, u::Union{AbstractArray{T,2}, AbstractArray{CT,2}, VectorField2D})
    return A.lmap*(T(A.λ)*u)
end