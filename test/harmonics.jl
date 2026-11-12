function setup_spectral(choices, params, sph; hd_n=8, hd_nu=1e-2,
                        mgr=MultiThread(VectorizedCPU()))
    case = choices.TestCase(Float64)
    params = merge(choices, case.params, params)
    gas = params.Fluid(params)
    vcoord = SigmaCoordinate(params.nz, params.ptop)
    surface_geopotential(lon, lat) = initial(case, lon, lat)[2]
    model = HPE(params, mgr, sph, vcoord, surface_geopotential, gas)
    state = let
        init(lon, lat) = initial(case, lon, lat)
        init(lon, lat, p) = initial(case, lon, lat, p)
        CFHydrostatics.initial_HPE(init, model)
    end
    diags = diagnostics(model)
    return model, state, diags
end

function harmonics()
    choices = (Fluid=IdealPerfectGas,
               consvar=:temperature,
               TestCase=Jablonowski06,
               precision=Float64,
               TimeScheme=RungeKutta4, # KinnmarkGray{2,5}
               nz=30, # max(30,4*Threads.nthreads()),
               nlat=96)

    params = (ptop=100,
              Cp=1000,
              kappa=2 / 7,
              p0=1e5,
              T0=300,
              radius=6.4e6,
              Omega=7.272e-5,
              courant=1.8,
              interval=6 * 3600)

    z = zero(choices.precision)
    pinthreads(:cores)

    threadinfo()

    scaling_pressure(choices, params)

    @info "Initializing spherical harmonics..."
    @time sph = SHTnsSphere(choices.nlat)

    @info "Spectral model setup..." sph choices.nz
    params = map(Float64, params)
    params = (Uplanet=params.radius * params.Omega, params...)
    @showtime spmodel, state0, diags = setup_spectral(choices, params, sph; mgr=PlainCPU())

    @twice dstate, scratch = CFHydrostatics.HPE_tendencies!(void, void, spmodel, sph,
                                                            state0, z)
    @twice dstate, scratch = CFHydrostatics.HPE_tendencies!(dstate, scratch, spmodel,
                                                            sph, state0, z)

    @testset "Spectral diagnostics" begin
        session = open(diags; model=spmodel, state=state0)
        @test session.surface_pressure ≈ dropdims(sum(session.masses.air; dims=3); dims=3) .+ params.ptop
        @test session.geopotential[:, :, 1] == spmodel.Phis
        @test size(session.Phi_dot) == (sph.nlat, sph.nlon, choices.nz)
        @test all(isfinite, session.Phi_dot)
    end

    @info "Spectral model time integration" choices.TimeScheme
    scheme = choices.TimeScheme(spmodel)
    scratch_scheme = CFTimeSchemes.scratch_space(scheme, state0, z)

    (; k0, k1, k2, k3) = scratch_scheme
    @twice future = CFTimeSchemes.advance!(void, scheme, state0, z, z, scratch_scheme)

    solver = IVPSolver(scheme, z)
    @twice future, t = CFTimeSchemes.advance!(void, solver, state0, z, 1)
    solver! = IVPSolver(scheme, z, state0, z) # mutating
    @twice future, t = CFTimeSchemes.advance!(future, solver!, state0, z, 1)

    nstep = 50
    @info "Spectral model Enzyme adjoint" nstep
    test_autodiff(Ext.Dynamics.tendencies!,
                  Dup(dstate), Dup(scratch), Const(spmodel), Dup(state0), Const(z))

    test_autodiff(CFTimeSchemes.advance!, Dup(future),
                  Const(scheme), Dup(state0), Const(z), Const(z), Dup(scratch_scheme))

    function repeat_advance!(sch, fut, scr)
        repeat(nstep, fut, scr, nothing) do st, scr, _
            CFTimeSchemes.advance!(st, sch, st, z, z, scr)
            return nothing # required !
        end
    end
    test_autodiff(repeat_advance!, Dup(scheme), Dup(future), Dup(scratch_scheme))
end

Dup(x) = Duplicated(x, make_zero(x))

function test_autodiff(fun, args...)
    @info fun
    vargs = map(x -> x.val, args)
    @twice fun(vargs...)
    # Broadcasts may mix constant model data with active state arrays.
    @twice autodiff(Enzyme.set_runtime_activity(Reverse), Const(fun), Const, args...)
end
