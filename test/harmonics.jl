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
               TimeScheme=KinnmarkGray{2,5}, # RungeKutta4,
               nz=30, # max(30,4*Threads.nthreads()),
               nlat=256)

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

    #    scaling_pressure(choices, params)

    @info "Initializing spherical harmonics..."
    @time sph = SHTnsSphere(choices.nlat)
    @info sph

    @info "Spectral model setup..."
    params = map(Float64, params)
    params = (Uplanet=params.radius * params.Omega, params...)
    @showtime spmodel, state0, diags = setup_spectral(choices, params, sph ; mgr=PlainCPU())

    @twice dstate, scratch = CFHydrostatics.HPE_tendencies!(void, void, spmodel, sph,
                                                               state0, z)
    @twice dstate, scratch = CFHydrostatics.HPE_tendencies!(dstate, scratch, spmodel,
                                                               sph, state0, z)

    @info "Spectral model time integration"
    scheme = choices.TimeScheme(spmodel)
    solver = IVPSolver(scheme, z)
    @twice future, t = CFTimeSchemes.advance!(void, solver, state0, z, 1)
    solver! = IVPSolver(scheme, z, state0, z) # mutating
    @twice future, t = CFTimeSchemes.advance!(future, solver!, state0, z, 1)

    @info "Spectral model Enzyme adjoint"
    if Base.VERSION >= v"1.10" && Base.VERSION < v"1.11"
        dup(x) = Duplicated(x, make_zero(x))
        state0_dup = dup(state0)
        dstate_dup = dup(dstate)
        scratch_dup = dup(scratch)

        tendencies! = Ext.Dynamics.tendencies!
        @twice tendencies!(dstate, scratch, spmodel, state0, z)
        @twice autodiff(Reverse, Const(tendencies!), Const, 
                            dstate_dup, scratch_dup, Const(spmodel), state0_dup, Const(z))
    else
        @warn "Enzyme is known to work only with Julia 1.10 at this time."
    end

end
