using Zygote
using ZygoteRules
using BSON: @save, @load
using NNlib
using LinearAlgebra: norm
using Random: seed!
using JLD2: save
using Statistics: mean
using ImageFiltering

include("xcat-loader-orig.jl")
include("bregem3.jl")
include("gen_plan.jl")
CUDA.allowscalar(false)

kernel = Float32.(Kernel.LoG((1,1,1)))
# Zygote does not support backpropagate through imfilter.
fLoG3d(x) = NNlib.conv(unsqueeze45(Array(x)), unsqueeze45(kernel), pad = 4)[:,:,:,1,1]
hfen(xout, xtrue) = norm(fLoG3d(xout) - fLoG3d(xtrue)) / norm(fLoG3d(xtrue))

# randseed = 2
shortcode = "3layer-end2end-xcat-3iter-1800-epochs"
train_path = "/media/myraid/data/SPECT-super-resolution/xcat/train/"
valid_path = "/media/myraid/data/SPECT-super-resolution/xcat/val/"
train_loader = xcatloader(train_path; shortcode)
valid_loader = xcatloader(valid_path; shortcode)
@assert size(train_loader.spect[1]) == (128, 128, 80)
@assert size(valid_loader.spect[1]) == (128, 128, 80)

seed!(0)
# cnn = Unet( ; init_filters = 4) |> gpu
cnn_list = []
nouter = 3

for i = 1:nouter
    cnn_cpu = Chain(
        Conv((3,3,3), 1 => 4, leakyrelu; stride = 1, pad = SamePad(), bias = true),
        Conv((3,3,3), 4 => 4, leakyrelu; stride = 1, pad = SamePad(), bias = true),
        Conv((3,3,3), 4 => 1; stride = 1, pad = SamePad(), bias = true),
        )
    # file = joinpath(pwd(), "ckpt/last-3layer-end2end-xcat-3iter-1500-epochs"*string(i)*".bson")
    # @load file cnn_cpu
    # cnn = cnn_cpu |> gpu
    push!(cnn_list, cnn)
end

nrmse(x, y) = norm(vec(x) - vec(y)) / norm(vec(y))
function loss(projectb, backprojectb, xtrue, ynoisy, r, Asum, xrecon, cnn_list, β; niter = 1, nouter = nouter)
    xout = bregem(projectb, backprojectb, ynoisy, r, Asum, xrecon, cnn_list[1], β; niter)
    for i = 1:nouter-1
        xout = bregem(projectb, backprojectb, ynoisy, r, Asum, xout, cnn_list[i+1], β; niter)
    end
    return nrmse(xout, xtrue)
end

nepoch = 1800
β = 1

A_train_list = []
Asum_train_list = []
for idx = 1:train_loader.num
    spect, xtrue, yi, ri, mumap, psf = grab_data(train_loader, idx)
    A, Asum = gen_plan(Array(mumap), Array(psf); T = eltype(mumap))
    push!(A_train_list, A)
    push!(Asum_train_list, Asum)
end

A_valid_list = []
Asum_valid_list = []
for idx = 1:valid_loader.num
    spect, xtrue, yi, ri, mumap, psf = grab_data(valid_loader, idx)
    A, Asum = gen_plan(Array(mumap), Array(psf); T = eltype(mumap))
    push!(A_valid_list, A)
    push!(Asum_valid_list, Asum)
end

@assert train_loader.num > 1 || throw("number of training data must be larger than 1")
train_loss = ones(nepoch, train_loader.num)
valid_loss = ones(nepoch, valid_loader.num)

println("*****start training!*****")
start_time = time()
for e = 1:nepoch
    println("epoch number: ", e)
    time0 = time()
    for idx = 1:train_loader.num # use the last data for validation
        for i = 1:nouter
            trainmode!(cnn_list[i])
        end
        spect, xtrue, yi, ri, mumap, psf = grab_data(train_loader, idx)
        A = A_train_list[idx]
        projectb(x) = CuArray(A * Array(x))
        @adjoint projectb(x) = CuArray(A * Array(x)), dy -> (CuArray(A' * Array(dy)), )

        backprojectb(y) = CuArray(A' * Array(y))
        @adjoint backprojectb(y) = CuArray(A' * Array(y)), dx -> (CuArray(A * Array(dx)), )

        train_loss[e, idx] = loss(projectb, backprojectb, xtrue, yi, ri,
                                  Asum_train_list[idx], spect, cnn_list, β; niter = 1)
        @show train_loss[e, idx]
        ps_list = []
        gs_list = []
        for i = 1:nouter
            ps = Flux.params(cnn_list[i])
            gs = gradient(ps) do
                loss(projectb, backprojectb, xtrue, yi, ri,
                     Asum_train_list[idx], spect, cnn_list, β; niter = 1)
            end
            push!(ps_list, ps)
            push!(gs_list, gs)
        end
        opt = ADAMW(0.002)
        # opt = Descent(0.002)
        for i = 1:nouter
            Flux.Optimise.update!(opt, ps_list[i], gs_list[i])
        end
    end
    println("*****start validation!*****")
    for idx = 1:valid_loader.num
        for i = 1:nouter
            testmode!(cnn_list[i])
        end
        spect, xtrue, yi, ri, mumap, psf = grab_data(valid_loader, valid_loader.num)
        A = A_valid_list[idx]
        projectb(x) = CuArray(A * Array(x))
        @adjoint projectb(x) = CuArray(A * Array(x)), dy -> (CuArray(A' * Array(dy)), )

        backprojectb(y) = CuArray(A' * Array(y))
        @adjoint backprojectb(y) = CuArray(A' * Array(y)), dx -> (CuArray(A * Array(dx)), )

        valid_loss[e, idx] = loss(projectb, backprojectb, xtrue, yi, ri, Asum_valid_list[idx],
                                  spect, cnn_list, β; niter = 1)
        @show valid_loss[e, idx]
        @show time() - time0 # 24 seconds, Effective GPU memory usage: 25.31% (5.993 GiB/23.678 GiB)
    end

    if e == argmin(vec(mean(valid_loss, dims=2)))
        println("The best CNN!")
        for i = 1:nouter
            file = "./ckpt/best-"*shortcode*string(i)*".bson" # adjust path/name as needed
            cnn_cpu = cnn_list[i] |> cpu
            @save file cnn_cpu # needs to be on cpu to save ckpt
        end
    end
    for i = 1:nouter
        file = "./ckpt/last-"*shortcode*string(i)*".bson" # adjust path/name as needed
        cnn_cpu = cnn_list[i] |> cpu
        @save file cnn_cpu # needs to be on cpu to save ckpt
    end
    GC.gc() # clean up memory
end
end_time = time()
@show end_time - start_time

save("./losses/train_loss_"*shortcode*".jld2", Dict("loss" => train_loss))
save("./losses/valid_loss_"*shortcode*".jld2", Dict("loss" => valid_loss))

# for e = 1:nepoch
#     time0 = time()
#     for (spect, xtrue, yi, ri, mumap, psf) in train_loader
#         spect = spect[:,:,:,1] # nx,ny,nz
#         xtrue = xtrue[:,:,:,1] # nx,ny,nz
#         yi = yi[:,:,:,1] # nx,nz,nview
#         ri = ri[:,:,:,1] # nx,nz,nview
#         mumap = mumap[:,:,:,1] # nx,ny,nz
#         psf = psf[:,:,:,:,1] # px,pz,ny,nview
#         A, Asum = Cugen_plan(mumap, psf; T = eltype(mumap))
#         projectb1(x) = A * x
#         @adjoint projectb1(x) = A * x, dy -> (A' * dy, )
#
#         backprojectb1(y) = A' * y
#         @adjoint backprojectb1(y) = A' * y, dx -> (A * dx, )
#         ps = Flux.params(cnn)
#         gs = gradient(ps) do
#             loss(projectb1, backprojectb1, xtrue, yi, ri, Asum, spect, cnn, β; niter = 1)
#         end
#         opt = ADAMW(0.002)
#         Flux.Optimise.update!(opt, ps, gs)
#         @show time() - time0 # 220 seconds, Effective GPU memory usage: 47.52% (11.252 GiB/23.678 GiB)
#     end
# end
# GC.gc(true)
# CUDA.memory_status()
