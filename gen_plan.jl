# gen_plan.jl
using LinearMapsAA
using LinearMaps
using SPECTrecon
using CUDA

function gen_plan(mumap::AbstractArray,
                  psfs::AbstractArray;
                  T::DataType = Float32)
    dy = 4.7952 # transaxial pixel size in mm
    nx, ny, nz = size(mumap)
    nview = size(psfs, 4)
    plan = SPECTplan(mumap, psfs, dy; T)
    idim = (nx,ny,nz)
    odim = (nx,nz,nview)
    A = LinearMapAA(x -> project(x, plan), y -> backproject(y, plan),
                   (prod(odim), prod(idim)); T, odim, idim)
    Asum = CuArray(A' * ones(T, nx, nz, nview))
    return A, Asum
end

# mumap = CuArray(rand(Float32,128,128,80))
# psfs = CuArray(ones(37,37,128,128))
# dy = 4.7952 # transaxial pixel size in mm
# nx, ny, nz = size(mumap)
# nview = size(psfs, 4)
# T = Float32
# plan = CuSPECTplan(mumap, psfs, dy; T)
# idim = (nx,ny,nz)
# odim = (nx,nz,nview)
# A = LinearMapAA(x -> Cuproject(x, plan), y -> Cubackproject(y, plan),
#                (prod(odim), prod(idim)); T, odim, idim)
# Asum = A' * CuArray(ones(T, nx, nz, nview))
# Asum1 = Cubackproject(CuArray(ones(T, nx, nz, nview)), plan)
#
#
# nx = 8
# ny = 8
# nz = 6
# idim = (nx,ny,nz)
# odim = (nx,ny,nz)
# T = Float32
# A = LinearMapAA(x -> 2 * x, y -> 2 * y, (prod(odim), prod(idim)); T, odim, idim)
# B = LinearMap(x -> 2 * x, y -> 2 * y, prod(odim))
# A * CuArray(ones(T, nx, ny, nz))
# B * CuArray(ones(T, prod(odim)))
