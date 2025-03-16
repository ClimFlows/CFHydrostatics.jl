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
               Prec=Float64,
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

    pinthreads(:cores)

    threadinfo()

    scaling_pressure(choices, params)

    @info "Initializing spherical harmonics..."
    @time sph = SHTnsSphere(choices.nlat)
    @info sph

    @info "Spectral model setup..."

    let params = map(Float64, params)
        params = (Uplanet=params.radius * params.Omega, params...)
        @time model, state0, diags = setup_spectral(choices, params, sph)
    end
end
