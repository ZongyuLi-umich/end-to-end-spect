using Zygote
using ZygoteRules
using BSON: @save, @load
using NNlib
using Flux
using LinearAlgebra: norm
using Random: seed!
using FileIO
using AVSfldIO
using Statistics: mean
using JLD2: save
using ImageFiltering
using MAT

include("xcat-loader-orig.jl")
include("bregem3.jl")
include("gen_plan.jl")
CUDA.allowscalar(false)
# randseed = 2
shortcode = "3layer-bcd-xcat-3iter-1800-epochs"
nrmse(x, y) = norm(vec(x) - vec(y)) / norm(vec(y))
kernel = Float32.(Kernel.LoG((1,1,1)))
# Zygote does not support backpropagate through imfilter.
fLoG3d(x) = NNlib.conv(unsqueeze45(Array(x)), unsqueeze45(kernel), pad = 4)[:,:,:,1,1]
hfen(xout, xtrue) = norm(fLoG3d(xout) - fLoG3d(xtrue)) / norm(fLoG3d(xtrue))
nepoch = 600
β = 1

function loss(xrecon, xtrue, cnn)
    xout = cnn(unsqueeze45(xrecon))[:,:,:,1,1] + xrecon
    return nrmse(xout, xtrue)
end

function regem_nrmse(xtrue, A, yi, ri, Asum, spect, cnn, β; niter = 1)
    xout_regem = regem(A, yi, ri, Asum, spect, cnn, β; niter = 1)
    return nrmse(xout_regem, xtrue)
end

function regem_hfen(xtrue, A, yi, ri, Asum, spect, cnn, β; niter = 1)
    xout_regem = regem(A, yi, ri, Asum, spect, cnn, β; niter = 1)
    return hfen(xout_regem, xtrue)
end

function forw(xrecon, xtrue, cnn)
    xout = cnn(unsqueeze45(xrecon))[:,:,:,1,1] + xrecon
    return xout
end

train_path = "/media/myraid/data/SPECT-super-resolution/xcat/train/"
valid_path = "/media/myraid/data/SPECT-super-resolution/xcat/val/"
test_path = "/media/myraid/data/SPECT-super-resolution/xcat/test/"
writepath = "/media/myraid/data/SPECT-super-resolution/xcat/"

start_time = time()
for layer = 1:3
    
    seed!(0)
    println("layer number: ", layer)
    train_loader = xcatloader(train_path; layer = layer-1, shortcode)
    valid_loader = xcatloader(valid_path; layer = layer-1, shortcode)
    @assert size(train_loader.spect[1]) == (128, 128, 80)
    # initialize CNN
    cnn_cpu = Chain(
        Conv((3,3,3), 1 => 4, leakyrelu; stride = 1, pad = SamePad(), bias = true),
        Conv((3,3,3), 4 => 4, leakyrelu; stride = 1, pad = SamePad(), bias = true),
        Conv((3,3,3), 4 => 1; stride = 1, pad = SamePad(), bias = true),
        )
    file = joinpath(pwd(), "ckpt/last-3layer-bcd-xloss-layer"*string(layer)*".bson")
    @load file cnn_cpu
    cnn = cnn_cpu |> gpu
    @assert train_loader.num > 1 || throw("number of training data must be larger than 1")
    train_loss = ones(nepoch, train_loader.num)
    valid_loss = ones(nepoch, valid_loader.num)
    train_loss_x = ones(nepoch, train_loader.num)
    valid_loss_x = ones(nepoch, valid_loader.num)
    println("*****start training!*****")
    for e = 1:nepoch
        println("epoch number: ", e)
        time0 = time()
        for idx = 1:train_loader.num # use the last data for validation
            trainmode!(cnn)
            spect, xtrue, yi, ri, mumap, psf = grab_data(train_loader, idx)
            A, Asum = gen_plan(Array(mumap), Array(psf); T = eltype(mumap))
            train_loss[e, idx] = loss(spect, xtrue, cnn)
            @show train_loss[e, idx]
            ps = Flux.params(cnn)
            gs = gradient(ps) do
                loss(spect, xtrue, cnn)
            end
            opt = ADAMW(0.002)
            Flux.Optimise.update!(opt, ps, gs)
            train_loss_x[e, idx] = regem_nrmse(xtrue, A, yi, ri, Asum, spect, cnn, β; niter = 1)
            @show train_loss_x[e, idx]
        end
        for idx = 1:valid_loader.num
            testmode!(cnn)
            spect, xtrue, yi, ri, mumap, psf = grab_data(valid_loader, idx)
            A, Asum = gen_plan(Array(mumap), Array(psf); T = eltype(mumap))
            valid_loss[e, idx] = loss(spect, xtrue, cnn)
            valid_loss_x[e, idx] = regem_nrmse(xtrue, A, yi, ri, Asum, spect, cnn, β; niter = 1)
            @show valid_loss[e, idx]
            @show valid_loss_x[e, idx]
            @show time() - time0 # 24 seconds, Effective GPU memory usage: 25.31% (5.993 GiB/23.678 GiB)
        end
        cnn_cpu = cnn |> cpu
        if e == argmin(vec(mean(valid_loss_x, dims=2)))
            println("The best CNN!")
            file = "./ckpt/best-"*shortcode*"-layer"*string(layer)*".bson" # adjust path/name as needed
            @save file cnn_cpu # needs to be on cpu to save ckpt
        end
        file = "./ckpt/last-"*shortcode*"-layer"*string(layer)*".bson" # adjust path/name as needed
        @save file cnn_cpu # needs to be on cpu to save ckpt
        GC.gc() # clean up memory
    end


    save("./losses/train_loss_"*shortcode*"-layer"*string(layer)*".jld2", Dict("loss" => train_loss))
    save("./losses/valid_loss_"*shortcode*"-layer"*string(layer)*".jld2", Dict("loss" => valid_loss))
    save("./losses/train_loss_x_"*shortcode*"-layer"*string(layer)*".jld2", Dict("loss" => train_loss_x))
    save("./losses/valid_loss_x_"*shortcode*"-layer"*string(layer)*".jld2", Dict("loss" => valid_loss_x))

    println("*****start testing!*****")
    test_loader = xcatloader(test_path; layer = layer-1, shortcode)
    @assert size(test_loader.spect[1]) == (128, 128, 80)
    cnn_cpu = Chain(
            Conv((3,3,3), 1 => 4, leakyrelu; stride = 1, pad = SamePad(), bias = true),
            Conv((3,3,3), 4 => 4, leakyrelu; stride = 1, pad = SamePad(), bias = true),
            Conv((3,3,3), 4 => 1; stride = 1, pad = SamePad(), bias = true),
        )

    # file = joinpath(pwd(), "ckpt/seed=$randseed"*"best-"*shortcode*"-layer"*string(layer)*".bson") # adjust path/name as needed
    file = joinpath(pwd(), "ckpt/best-3layer-bcd-layer"*string(layer)*".bson")
    @load file cnn_cpu
    cnn = cnn_cpu |> gpu
    for idx = 1:test_loader.num
        spect, xtrue, yi, ri, mumap, psf = grab_data(test_loader, idx)
        A, Asum = gen_plan(Array(mumap), Array(psf); T = eltype(mumap))
        testmode!(cnn)
        xout_cnn = forw(spect, xtrue, cnn)
        foldername = split(test_loader.filename[idx], ".")[1]
        # @show joinpath(writepath, "test", foldername, "seed=$randseed"*"cnn-"*shortcode*"-layer"*string(layer)*test_loader.filename[idx])
        # save(joinpath(writepath, "test", foldername,
        #      "cnn-"*shortcode*"-layer"*string(layer)*test_loader.filename[idx]*".fld"),
        #      Array(xout_cnn))
        matwrite(joinpath(writepath, "test", foldername,
                "cnn-"*shortcode*"-layer"*string(layer)*test_loader.filename[idx]),
                Dict("x" => Array(xout_cnn)))
        xout_regem = regem(A, yi, ri, Asum, spect, cnn, β; niter = 1)
        # save(joinpath(writepath, "test", foldername,
        #      "regem-"*shortcode*"-layer"*string(layer)*test_loader.filename[idx]*".fld"),
        #      Array(xout_regem))
        matwrite(joinpath(writepath, "test", foldername,
                "regem-"*shortcode*"-layer"*string(layer)*test_loader.filename[idx]),
                Dict("x" => Array(xout_regem)))
    end

    # generate training data for next iteration
    for idx = 1:train_loader.num
        spect, xtrue, yi, ri, mumap, psf = grab_data(train_loader, idx)
        A, Asum = gen_plan(Array(mumap), Array(psf); T = eltype(mumap))
        testmode!(cnn)
        xout_regem = regem(A, yi, ri, Asum, spect, cnn, β; niter = 1)
        foldername = split(train_loader.filename[idx], ".")[1]
        # save(joinpath(writepath, "train", foldername,
        #      "regem-"*shortcode*"-layer"*string(layer)*train_loader.filename[idx]*".fld"),
        #      Array(xout_regem))
        matwrite(joinpath(writepath, "train", foldername,
                "regem-"*shortcode*"-layer"*string(layer)*train_loader.filename[idx]),
                Dict("x" => Array(xout_regem)))
    end

    # generate valid data for next iteration
    for idx = 1:valid_loader.num
        spect, xtrue, yi, ri, mumap, psf = grab_data(valid_loader, idx)
        A, Asum = gen_plan(Array(mumap), Array(psf); T = eltype(mumap))
        testmode!(cnn)
        xout_regem = regem(A, yi, ri, Asum, spect, cnn, β; niter = 1)
        foldername = split(valid_loader.filename[idx], ".")[1]
        # save(joinpath(writepath, "val", foldername,
        #      "regem-"*shortcode*"-layer"*string(layer)*valid_loader.filename[idx]*".fld"),
        #      Array(xout_regem))
        matwrite(joinpath(writepath, "val", foldername,
                "regem-"*shortcode*"-layer"*string(layer)*valid_loader.filename[idx]),
                Dict("x" => Array(xout_regem)))
    end
end

end_time = time()
@show end_time - start_time





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
