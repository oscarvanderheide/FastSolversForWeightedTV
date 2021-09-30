using BenchmarkTools, Flux, Random
Random.seed!(1)

function fun1(ptn::AbstractArray{T,2}, ε::T) where T
    ptn = sort(vec(ptn))
    Σptn = cumsum(ptn[end:-1:1])[end:-1:1]-(length(ptn):-1:1).*ptn
    val = Σptn .<= ε
    findfirst(val)
end

function fun2(ptn::AbstractArray{T,2}, ε::T) where T
    ptn = sort(vec(ptn))
    Σptn = cumsum(ptn[end:-1:1])[end:-1:1]-(length(ptn):-1:1).*ptn
    # @time val = Σptn .<= ε
    searchsortedfirst(Σptn[end:-1:1], ε)
end

y = randn(Float32, 256, 256, 2) |> gpu
ptn = sqrt.(y[:,:,1].^2+y[:,:,2].^2)
ε = 0.1f0*sum(ptn)

@btime fun1(ptn, ε)
@btime fun2(ptn, ε)