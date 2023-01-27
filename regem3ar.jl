# bregem.jl
using Flux
using LinearMapsAA: LinearMapAO
using CUDA: CuArray
using Zygote: @nograd
# using ChainRulesCore.@non_differentiable

nogradterms(A, y, x, r) = CuArray(A' * Array(y ./ (CuArray(A * Array(x)) .+ r)))
Zygote.@nograd nogradterms

nogradterms_esr(A, T, y, x, r) = CuArray(T' * (A' * Array(y ./ (CuArray(A * (T * Array(x))) .+ r))))
Zygote.@nograd nogradterms_esr

function unsqueeze45(x)
    return Flux.unsqueeze(Flux.unsqueeze(x, 4), 5)
end

function permute4(x)
    return permutedims(Flux.unsqueeze(x, 4), [1,2,4,3])
end

function unpermute4(x)
    return permutedims(x, [1,2,4,3])[:,:,:,1]
end


function regem(A::LinearMapAO,
               y::AbstractArray,
               r::AbstractArray,
               Asum::AbstractArray,
               x::AbstractArray,
               cnn::Any,
               β::Real;
               niter::Int = 1)

    u = cnn(unsqueeze45(x))[:,:,:,1,1] + x # residual learning
    # u = unpermute4(cnn(permute4(x))) + x # residual learning
    Asumu = Asum - β * u
    Asumu2 = Asumu.^2
    T = eltype(x)
    for iter = 1:niter
        eterm = nogradterms(A, y, x, r)
        eterm_beta = 4 * β * (x .* eterm)
        x = max.(0, T(1/2β) * (-Asumu + sqrt.(Asumu2 + eterm_beta)))
    end
    return x
end


function regemdown(A::LinearMapAO,
                   T::LinearMapAO,
                   y::AbstractArray,
                   r::AbstractArray,
                   TAsum::AbstractArray,
                   x::AbstractArray,
                   cnn::Any,
                   β::Real;
                   niter::Int = 1)

    u = cnn(unsqueeze45(x))[:,:,:,1,1] + x # residual learning
    # u = unpermute4(cnn(permute4(x))) + x # residual learning
    TAsumu = TAsum - β * u
    TAsumu2 = TAsumu.^2
    Ty = eltype(x)
    for iter = 1:niter
        eterm = nogradterms_esr(A, T, y, x, r)
        eterm_beta = 4 * β * (x .* eterm)
        x = max.(0, Ty(1/2β) * (-TAsumu + sqrt.(TAsumu2 + eterm_beta)))
    end
    return x
end
