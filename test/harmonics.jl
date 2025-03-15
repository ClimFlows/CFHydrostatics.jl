function setup_spectral(choices, params, sph; hd_n = 8, hd_nu = 1e-2, mgr = MultiThread(VectorizedCPU()))
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
