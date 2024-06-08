using LoopManagers: PlainCPU, VectorizedCPU, MultiThread
using MutatingOrNot: void
using SHTnsSpheres: SHTnsSphere, synthesis_scalar!
using ClimFlowsTestCases: testcase, initial_surface, initial_flow, Jablonowski06
using ClimFluids: IdealPerfectGas
using CFDomains: SigmaCoordinate

using CFHydrostatics: CFHydrostatics, HPE, diagnostics
using ThreadPinning
using Test

const Ext = Base.get_extension(CFHydrostatics, :SHTnsSpheres_Ext)
const hydrostatic_pressure! = Ext.Dynamics.hydrostatic_pressure!

function setup(choices, params, sph; hd_n = 8, hd_nu = 1e-2, mgr = MultiThread(VectorizedCPU()))
    case = testcase(choices.TestCase, Float64)
    params = merge(choices, case.params, params)
    gas = params.Fluid(params)
    vcoord = SigmaCoordinate(params.nz, params.ptop)
    surface_geopotential(lon, lat) = initial_surface(lon, lat, case)[2]
    model = HPE(params, mgr, sph, vcoord, surface_geopotential, gas)
    state = let
        init(lon, lat) = initial_surface(lon, lat, case)
        init(lon, lat, p) = initial_flow(lon, lat, p, case)
        CFHydrostatics.initial_HPE(init, model)
    end
    diags = diagnostics(model)
    return model, state, diags
end

function time(fun, N)
    fun()
    times = [(@timed fun()).time for i=1:(N+10)]
    sort!(times)
    return Float32(sum(times[1:N])/N)
end

function scaling(fun, name, simd, N::Int)
    @info "Multithread scaling for $name with $simd"
    @info "Threads \t elapsed \t speedup \t efficiency"
    single = 1e9
    for nt=1:Threads.nthreads()
        mgr = MultiThread(simd, nt)
        elapsed = time(N) do
            fun(mgr)
        end
        nt==1 && (single = elapsed)
        speedup = single/elapsed
        percent(x) = round(100*x; digits=0)
        @info "$nt \t\t $elapsed \t $(percent(speedup)) \t\t $(percent(speedup/nt))"
    end
end

function scaling_pressure(choices)
    model(mgr) = (; mgr, planet=(; radius=1.0), vcoord=(; ptop=1e3))
    mass = randn(2*choices.nlat, choices.nlat, choices.nz, 2)
    p = hydrostatic_pressure!(void, model(PlainCPU()), mass)
    for vsize in (16,32)
        scaling("hydrostatic_pressure!", VectorizedCPU(vsize), 100) do mgr
            hydrostatic_pressure!(p, model(mgr), mass)
        end
    end
end

choices = (
    Fluid = IdealPerfectGas,
    consvar = :temperature,
    TestCase = Jablonowski06,
    Prec = Float64,
    nz = max(30,4*Threads.nthreads()),
    nlat = 128
)
params = (
    ptop = 100,
    Cp = 1000,
    kappa = 2 / 7,
    p0 = 1e5,
    T0 = 300,
    radius = 6.4e6,
    Omega = 7.272e-5,
    courant = 1.8,
    interval = 6 * 3600, # 6-hour intervals
)

pinthreads(:cores)

threadinfo()

scaling_pressure(choices)

exit()

@info "Initializing spherical harmonics..."
@time sph = SHTnsSphere(choices.nlat)
@info sph

@info "Model setup..."

params = map(Float64, params)
params = (Uplanet = params.radius * params.Omega, params...)
@time model, state0, diags = setup(choices, params, sph)

display(methods(typeof(model)))
        
exit()

@info "time for 100 scalar spectral transforms"
@time begin
    for _ in 1:100
        synthesis_scalar!(spat, spec, sph)
    end
end

@testset "CFHydrostatics.jl" begin
    simd = VectorizedCPU()    
end
