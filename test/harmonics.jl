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

    @showtime dstate, scratch = CFHydrostatics.HPE_tendencies!(void, void, spmodel, sph,
                                                               state0, z)
    @showtime dstate, scratch = CFHydrostatics.HPE_tendencies!(dstate, scratch, spmodel,
                                                               sph, state0, z)
    @showtime dstate, scratch = CFHydrostatics.HPE_tendencies!(dstate, scratch, spmodel,
                                                               sph, state0, z)

    @info "Spectral model time integration"
    scheme = choices.TimeScheme(spmodel)
    solver = IVPSolver(scheme, z)
    @showtime future, t = CFTimeSchemes.advance!(void, solver, state0, z, 1)
    @showtime future, t = CFTimeSchemes.advance!(void, solver, state0, z, 1)
    solver! = IVPSolver(scheme, z, state0, z) # mutating
    @showtime future, t = CFTimeSchemes.advance!(future, solver!, state0, z, 1)
    @showtime future, t = CFTimeSchemes.advance!(future, solver!, state0, z, 1)

    @info "Spectral model adjoint"
    dup(x) = Duplicated(x, make_zero(x))
    state0_dup = dup(state0)
    dstate_dup = dup(dstate)
    scratch_dup = dup(scratch)

    hydrostatic_pressure! = Ext.Dynamics.hydrostatic_pressure!
    mass_budgets! = Ext.Dynamics.mass_budgets!
    Bernoulli! = Ext.Dynamics.Bernoulli!
    tendencies! = Ext.Dynamics.tendencies!_

    function fun(dstate, scratch, model, sph, state)
        (; locals, locals_dmass, locals_duv) = scratch
        (; uv, mass_air, mass_consvar, p, B, exner, consvar, geopot) = locals
        (; mass_air_spec, mass_consvar_spec, uv_spec) = state
        dmass_air_spec, dmass_consvar_spec, duv_spec = dstate

        fcov = model.fcov
        metric = model.planet.radius^-2
        #        sph = model.domain.layer

        # flux-form mass balance
        (; dmass_air_spec, dmass_consvar_spec, uv, mass_air, mass_consvar), locals_dmass = mass_budgets!((;
                                                                                                          dmass_air_spec,
                                                                                                          dmass_consvar_spec,
                                                                                                          uv,
                                                                                                          mass_air,
                                                                                                          mass_consvar),
                                                                                                         scratch.locals_dmass,
                                                                                                         (;
                                                                                                          mass_air_spec,
                                                                                                          mass_consvar_spec,
                                                                                                          uv_spec),
                                                                                                         sph,
                                                                                                         sph.laplace,
                                                                                                         metric)

        # hydrostatic balance, geopotential, exner function
        p = hydrostatic_pressure!(p, model, mass_air)
        B, exner, consvar, geopot = Bernoulli!((B, exner, consvar, geopot),
                                               (mass_air, mass_consvar, p, uv), model)
        return nothing
    end
    #    mode = set_runtime_activity(Reverse)
    mode = Reverse

    @twice fun(dstate, scratch, spmodel, sph, state0) # forward model
    @twice Enzyme.autodiff(mode, Const(fun), Const, dstate_dup, scratch_dup, Const(spmodel),
                           Const(sph),
                           state0_dup)
    @twice tendencies!(dstate, scratch, spmodel, sph, state0) # forward model
    @twice Enzyme.autodiff(mode, Const(tendencies!), Const, dstate_dup, scratch_dup,
                           Const(spmodel), Const(sph),
                           state0_dup)

    @info "Spectral model adjoint + time integration"
    #=
    square(x::NamedTuple) = mapreduce(square, +, x)
    square(x::Array) = sum(x.*x)
    function loss(dstate, scratch, present)
        dstate, scratch = CFHydrostatics.HPE_tendencies!(dstate, scratch, model, sph, state0, z)
        #        future, t = CFTimeSchemes.advance!(future, solver!, present, z, 1)
        return square(dstate)
    end
    @showtime loss(dtsate, scratch, state0)
    =#

end
