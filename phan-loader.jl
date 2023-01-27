# phan-loader.jl
# load xcat data
using CUDA
using MAT: matread
using FileIO
using AVSfldIO

struct phanloader{T}
    ```
    -`num`: number of data
    -`spect`: OSEM reconstruction image using 16 iterations and 4 subsets
    -`xtrue`: True activity map
    -`yi`: Total projections
    -`ri`: Scatters
    -`mumap`: Attenuation map
    -`psf`: Point spread function
    ```
    num::Int
    spect::Vector{CuArray{T, 3}}
    xtrue::Vector{CuArray{T, 3}}
    yi::Vector{CuArray{T, 3}}
    ri::Vector{CuArray{T, 3}}
    mumap::Vector{CuArray{T, 3}}
    psf::Vector{CuArray{T, 4}}
    filename::Vector{String}
    function phanloader(path::String; T::DataType = Float32,
                        layer = 0, shortcode = "bcd", randseed=1)
        spect = Array{T, 3}[]
        xtrue = Array{T, 3}[]
        yi = Array{T, 3}[]
        ri = Array{T, 3}[]
        mumap = Array{T, 3}[]
        psf = Array{T, 4}[]
        filename = String[]
        if layer > 0
            spectstr = "seed=$randseed"*"regem-"*shortcode*"-layer"*string(layer)
        else
            spectstr = "seed=$randseed"*"_osem16"
        end
        xtruestr = "seed=$randseed"*"_xtrue_scaled"
        yistr = "seed=$randseed"*"_ynoisy"
        ristr = "seed=$randseed"*"_background"
        mumapstr = "mumap"
        psfstr = "psf"
        for (root, dir, files) in walkdir(path)
            # println("Directories in "*string(root))
            for file in files
                if occursin(spectstr, file)
                    println("load spect from "*joinpath(root, file)) # path to files
                    push!(spect, T.(load(joinpath(root, file))))
                end
                if occursin(xtruestr, file)
                    println("load xtrue from "*joinpath(root, file)) # path to files
                    push!(xtrue, T.(load(joinpath(root, file))))
                end
                if occursin(yistr, file)
                    println("load yi from "*joinpath(root, file)) # path to files
                    push!(yi, T.(load(joinpath(root, file))))
                end
                if occursin(ristr, file)
                    println("load ri from "*joinpath(root, file)) # path to files
                    push!(ri, T.(load(joinpath(root, file)) .+ 1e-8))
                end
                if occursin(mumapstr, file)
                    println("load mumap from "*joinpath(root, file)) # path to files
                    push!(mumap, T.(load(joinpath(root, file))))
                    push!(filename, String(split(file, "_")[end]))
                end
                if occursin(psfstr, file)
                    println("load psf from "*joinpath(root, file)) # path to files
                    push!(psf, T.(load(joinpath(root, file))))
                end
            end
        end
        nf = length(spect)
        @assert nf > 0 || throw("empty spect!")
        (nx, ny, nz) = size(spect[1])
        @assert length(psf) > 0 || throw("empty psf!")
        (px, pz, _, nview) = size(psf[1])
        @assert length(mumap) > 0 || throw("empty mumap!")
        (dx, dy, dz) = size(mumap[1])

        for i = 1:length(spect)
            @show extrema(spect[i])
            spect[i] = CuArray(spect[i])
            @show extrema(xtrue[i])
            xtrue[i] = CuArray(xtrue[i])
            @show extrema(yi[i])
            yi[i] = CuArray(yi[i])
            @show extrema(ri[i])
            ri[i] = CuArray(ri[i])
            @show extrema(mumap[i])
            mumap[i] = CuArray(mumap[i])
            @show extrema(psf[i])
            psf[i] = CuArray(psf[i])
        end
        num = length(spect)
        new{T}(num, spect, xtrue, yi, ri, mumap, psf, filename)
    end
end

function grab_data(loader::phanloader, idx::Int)
    @assert (idx>0 && idx < loader.num+1) || throw("invalid idx!")
    return ((@view loader.spect[idx])[1],
            (@view loader.xtrue[idx])[1],
            (@view loader.yi[idx])[1],
            (@view loader.ri[idx])[1],
            (@view loader.mumap[idx])[1],
            (@view loader.psf[idx])[1])
end

# spect, xtrue, yi, ri, mumap, psf = grab_data(train_loader, 1)
