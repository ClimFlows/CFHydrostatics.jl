# computing
using KernelAbstractions, Adapt, ManagedLoops, LoopManagers
using ManagedLoops: synchronize, @with, @vec, @unroll
using SIMDMathFunctions
using LoopManagers: LoopManager, PlainCPU, VectorizedCPU, MultiThread, no_simd
using MutatingOrNot: void, similar!
using ThreadPinning

using Pkg, Test, UUIDs

oneAPI_functional = try
    using oneAPI
    oneAPI.functional()
catch e
    false
end

CUDA_functional = try
    using CUDA
    CUDA.functional()
catch e
    false
end

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

harmonics()

voronoi()

exit()

@info "time for 100 scalar spectral transforms"
@time begin
    for _ in 1:100
        synthesis_scalar!(spat, spec, sph)
    end
end
