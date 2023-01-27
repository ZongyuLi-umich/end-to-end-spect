using Zygote
using ZygoteRules
using BSON: @load
using Random: seed!
using NNlib
using FileIO
using AVSfldIO

include("phan-loader.jl")
include("regem3ar.jl")
include("gen_plan.jl")
CUDA.allowscalar(false)

test_path = "/media/myraid/data/SPECT-super-resolution/xcat/test/" # change path as needed
randseed = 2
shortcode = "3layer-ar-xcat-3iter-seed=$randseed"
test_loader = phanloader(test_path; shortcode, randseed)
@assert size(test_loader.spect[1]) == (128, 128, 80)

seed!(0)
# cnn_cpu = Unet( ; init_filters = 4)

cnn_list = []
nouter = 3


for i = 1:nouter
    cnn_cpu = Chain(
                Conv((3,3,3), 1 => 4, leakyrelu; stride = 1, pad = SamePad(), bias = true),
                Conv((3,3,3), 4 => 4, leakyrelu; stride = 1, pad = SamePad(), bias = true),
                Conv((3,3,3), 4 => 1; stride = 1, pad = SamePad(), bias = true),
                )
    # file = joinpath(pwd(), "ckpt/seed=$randseed"*"best-"*shortcode*string(i)*".bson") # adjust path/name as needed
    file = joinpath(pwd(), "ckpt/best-3layer-ar-diff"*string(i)*".bson")
    @load file cnn_cpu
    cnn = cnn_cpu |> gpu
    push!(cnn_list, cnn)
end

function forw(A, ynoisy, r, Asum, xrecon, cnn_list, β; niter = 1, nouter = nouter)
    xout = regem(A, ynoisy, r, Asum, xrecon, cnn_list[1], β; niter)
    for i = 1:nouter-1
        xout = regem(A, ynoisy, r, Asum, xout, cnn_list[i+1], β; niter)
    end
    return xout
end

β = 1
xout_list = []
writepath = "/media/myraid/data/SPECT-super-resolution/xcat/" # change path as needed

for idx = 1:test_loader.num
    spect, xtrue, yi, ri, mumap, psf = grab_data(test_loader, idx)
    A, Asum = gen_plan(Array(mumap), Array(psf); T = eltype(mumap))
    for i = 1:nouter
        testmode!(cnn_list[i])
    end
    xout = forw(A, yi, ri, Asum, spect, cnn_list, β; niter = 1)
    foldername = split(test_loader.filename[idx], ".")[1]
    save(joinpath(writepath, "test", foldername,
         "seed=$randseed"*"regem-"*shortcode*test_loader.filename[idx]*".fld"),
         Array(xout))
    push!(xout_list, xout)
end

pred_cnn = Array(xout_list[1])

using MIRTjim
spect, xtrue, yi, ri, mumap, psf = grab_data(test_loader, 1)
spect = Array(spect)
xtrue = Array(xtrue)
idx = 37
cmax = maximum(xtrue[:,:,idx])
clim = (0, cmax)
color =:viridis
jim(jim(spect[:,:,idx]; clim, color),
    jim(pred_cnn[:,:,idx]; clim, color),
    jim(xtrue[:,:,idx]; clim, color),
    jim(spect[:,:,idx] - pred_cnn[:,:,idx]; color),
    xlim = (1, 128))
