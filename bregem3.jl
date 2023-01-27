# bregem.jl
using Flux
using LinearMapsAA: LinearMapAO
using CUDA: CuArray

function unsqueeze45(x)
    return Flux.unsqueeze(Flux.unsqueeze(x, 4), 5)
end

function permute4(x)
    return permutedims(Flux.unsqueeze(x, 4), [1,2,4,3])
end

function unpermute4(x)
    return permutedims(x, [1,2,4,3])[:,:,:,1]
end

"""
    bregem(projectb, backprojectb, y, r, Asum, x, cnn, β; niter = 1)
Backpropagatable regularized EM reconstruction with CNN
-`projectb`: backpropagatable forward projection
-`backprojectb`: backpropagatable backward projection
-`y`: projections
-`r`: scatters
-`Asum`: A' * 1
-`x`: current iterate
-`cnn`: Any cnn
-`β`: Regularized parameter
-`niter`: number of iteration for EM
"""
function bregem(projectb::Function,
                backprojectb::Function,
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
        eterm = backprojectb((y ./ (projectb(x) .+ r)))
        eterm_beta = 4 * β * (x .* eterm)
        x = max.(0, T(1/2β) * (-Asumu + sqrt.(Asumu2 + eterm_beta)))
    end
    return x
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
        eterm = CuArray(A' * Array(y ./ (CuArray(A * Array(x)) .+ r)))
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
        eterm = CuArray(T' * (A' * Array(y ./ (CuArray(A * (T * Array(x))) .+ r))))
        eterm_beta = 4 * β * (x .* eterm)
        x = max.(0, Ty(1/2β) * (-TAsumu + sqrt.(TAsumu2 + eterm_beta)))
    end
    return x
end
