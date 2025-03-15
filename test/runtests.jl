# computing
using KernelAbstractions, Adapt, ManagedLoops, LoopManagers
using ManagedLoops: synchronize, @with, @vec, @unroll
using SIMDMathFunctions
using LoopManagers: LoopManager, PlainCPU, VectorizedCPU, MultiThread, no_simd
using MutatingOrNot: void, similar!
using ThreadPinning
using Test

# maths
using SHTnsSpheres: SHTnsSphere, synthesis_scalar!
using CFDomains: CFDomains, Stencils, VoronoiSphere, SigmaCoordinate
using CFTimeSchemes: CFTimeSchemes, RungeKutta4, KinnmarkGray, IVPSolver
using LinearAlgebra: mul!

# physics
using ClimFlowsTestCases: describe, initial, Jablonowski06
using ClimFluids: IdealPerfectGas
using CFHydrostatics: CFHydrostatics, HPE, diagnostics
using CFPlanets: ShallowTradPlanet

# data
using NetCDF: ncread, ncwrite, nccreate, ncclose
using ClimFlowsData: DYNAMICO_reader, DYNAMICO_meshfile
using PrettyTables

if CFHydrostatics.PackageExtensionCompat.HAS_NATIVE_EXTENSIONS
    const Ext = Base.get_extension(CFHydrostatics, :SHTnsSpheres_Ext)
    const hydrostatic_pressure! = Ext.Dynamics.hydrostatic_pressure!
    const Bernoulli! = Ext.Dynamics.Bernoulli!
else
    using CFHydrostatics.SHTnsSpheres_Ext.Dynamics: hydrostatic_pressure!, Bernoulli!
end

include("scaling.jl")
include("harmonics.jl")
include("voronoi_ops.jl")
include("voronoi.jl")

function spectral()
    function choices()
        return (Fluid=IdealPerfectGas,
                consvar=:temperature,
                TestCase=Jablonowski06,
                Prec=Float64,
                nz=30, # max(30,4*Threads.nthreads()),
                nlat=256)
    end

    function params()
        return (ptop=100,
                Cp=1000,
                kappa=2 / 7,
                p0=1e5,
                T0=300,
                radius=6.4e6,
                Omega=7.272e-5,
                courant=1.8,
                interval=6 * 3600)
    end

    pinthreads(:cores)

    threadinfo()

    scaling_pressure(choices)

    @info "Initializing spherical harmonics..."
    @time sph = SHTnsSphere(choices().nlat)
    @info sph

    @info "Spectral model setup..."

    let params = map(Float64, params())
        params = (Uplanet=params.radius * params.Omega, params...)
        @time model, state0, diags = setup_spectral(choices(), params, sph)
    end
end

# spectral()

choices = (gpu_blocks=(0, 32),
           precision=Float32,
           nz=(32, 96, 96 * 4),
           # numerics
           meshname=DYNAMICO_meshfile("uni.1deg.mesh.nc"),
           coordinate=SigmaCoordinate,
           consvar=:temperature,
           TimeScheme=KinnmarkGray{2,5}, # RungeKutta4,
           # physics
           Fluid=IdealPerfectGas,
           TestCase=Jablonowski06)

params = (
          # numerics
          courant=4.0,
          # physics
          radius=6.4e6,
          Omega=7.27220521664304e-5,
          ptop=225.52395239472398, # compatible with NCARL30 vertical coordinate
          Cp=1004.5,
          kappa=2 / 7,
          p0=1e5,
          T0=300,
          nu_gradrot=1e-16,
          hyperdiff_nu=0.002,
          # simulation
          testcase=(), # to override default test case parameters
          interval=6 * 3600)

reader = DYNAMICO_reader(ncread, choices.meshname)
vsphere = VoronoiSphere(reader; prec=choices.precision)
@info vsphere

mgrs = plain, simd = PlainCPU(), VectorizedCPU()
mgr_names = ["CPU", "SIMD"]

try
    global mgrs, mgr_names
    using oneAPI
    if oneAPI.functional()
        @info "Functional oneAPI GPU detected !"
        oneAPI.versioninfo()
        gpu0 = LoopManagers.KernelAbstractions_GPU(oneAPIBackend(), (0, 0))
        gpu = LoopManagers.KernelAbstractions_GPU(oneAPIBackend(), choices.gpu_blocks)
        mgrs = (mgrs..., gpu0, gpu)
        mgr_names = ["CPU", "SIMD", "GPU", "GPU(blocked)"]
    end
catch e
end

try
    global mgrs, mgr_names
    using CUDA
    if CUDA.functional()
        @info "Functional CUDA GPU detected !"
        CUDA.versioninfo()
        gpu0 = LoopManagers.KernelAbstractions_GPU(CUDABackend(), (0, 0))
        gpu = LoopManagers.KernelAbstractions_GPU(CUDABackend(), choices.gpu_blocks)
        mgrs = (mgrs..., gpu0, gpu)
        mgr_names = ["CPU", "SIMD", "GPU", "GPU(blocked)"]
    end
catch e
end

function voronoi()
    model, diagnostics(model), state = setup_voronoi(vsphere, (choices..., nz=choices.nz[1]), params, simd)

    for nz in choices.nz
        M = randn(choices.precision, 1024, 1024)
        N = randn(choices.precision, 1024, 1024)
        ue = randn(choices.precision, nz, length(edges(vsphere)))
        qi = randn(choices.precision, nz, length(cells(vsphere)))
        qv = randn(choices.precision, nz, length(duals(vsphere)))

        data = vcat(bench(vexp!, mgrs, M),
#                    bench(mmul!, mgrs, M, N),
#                    bench(gradient!, mgrs, vsphere, qi),
#                    bench(gradperp!, mgrs, vsphere, qv),
#                    bench(perp!, mgrs, vsphere, ue),
#                    bench(curl!, mgrs, vsphere, ue),
                    bench(divergence!, mgrs, vsphere, ue),
                    bench(TRiSK!, mgrs, vsphere, ue))

        header = (["nz=$nz", mgr_names...])
        best = Highlighter((data, i, j) -> j > 1 &&
                               all(data[i, k] >= data[i, j] for k in 2:size(data, 2)),
                           crayon"red bold")

        pretty_table(data;
                     header=header,
                     formatters=ft_printf("%7.6f", 2:5),
                     header_crayon=crayon"yellow bold",
                     highlighters=best,
                     tf=tf_unicode_rounded)
    end

    return GC.gc(true) # free GPU resources before exiting to avoid segfault ?
end

voronoi()

exit()

@info "time for 100 scalar spectral transforms"
@time begin
    for _ in 1:100
        synthesis_scalar!(spat, spec, sph)
    end
end
