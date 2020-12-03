# General utilities


using CUDA

use_gpu(X) = X isa CuArray
convert_cu(in_a, X) =  X isa CuArray ? cu(in_a) : in_a
cuzeros(X, args...) = X isa CuArray ? CUDA.zeros(args...) : zeros(T, args...)
cuones(X, args...) = X isa CuArray ? CUDA.ones(args...) : ones(T, args...)