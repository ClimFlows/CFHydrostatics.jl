"""
    state = initial_HPE(case, model)

Return an initial state for the hydrostatic `model`.
`case` is expected to be a function with two methods:

    surface_pressure, surface_geopotential = case(longitude, latitude)
    geopotential, ulon, ulat = case(longitude, latitude, pressure)

If the hydrostatic model is for a binary fluid, the second method should return instead:

    geopotential, ulon, ulat, q = case(longitude, latitude, pressure)
"""
initial_HPE(case, model) = initial_HPE(model, model.domain.layout, case)

initial_HPE(model, ::CFDomains.HVLayout, case) =
    initial_HPE_HV(model, CFDomains.nlayer(model.domain), model.domain.layer, case)

initial_HPE(model, ::CFDomains.VHLayout, case) =
    initial_HPE_VH(model, CFDomains.nlayer(model.domain), model.domain.layer, case)

# implemented in SHTnsSphere_Ext
function initial_HPE_HV end

# implemented in voronoi
function initial_HPE_VH end

function initial_HPE_HV_collocated(model, nz, lon, lat, gas::SimpleFluid, case)
    # mass[i, j, k, 1] = dry air mass
    # mass[i, j, k, 2] = mass-weighted conservative variable
    radius, vcoord = model.planet.radius, model.vcoord
    consvar = gas(:p, :v).conservative_variable
    alloc(dims...) = similar(lon, size(lon)..., dims...)
    masses, ulon, ulat = (air=alloc(nz), consvar=alloc(nz)), alloc(nz), alloc(nz)

    for i in axes(lon, 1), j in axes(lon, 2), k = 1:nz
        let lon = lon[i, j], lat = lat[i, j]
            ps, _ = case(lon, lat)
            p = pressure_level(2k - 1, ps, vcoord) # full level k
            _, uu, vv = case(lon, lat, p)
            ulon[i, j, k], ulat[i, j, k] = radius * uu, radius * vv
            p_lower = pressure_level(2k - 2, ps, vcoord) # lower interface
            p_upper = pressure_level(2k, ps, vcoord) # upper interface
            mg = p_lower - p_upper
            Phi_lower, _, _ = case(lon, lat, p_lower)
            Phi_upper, _, _ = case(lon, lat, p_upper)
            v = (Phi_upper - Phi_lower) / mg # dPhi = -v . dp
            masses.air[i, j, k] = radius^2 * mg
            masses.consvar[i, j, k] = (radius^2 * mg) * consvar(p, v)
        end
    end
    return masses, ulon, ulat
end

function initial_HPE_VH_collocated(model, nz, lon, lat, gas::SimpleFluid, case)
    # mass_air[k, ij] = dry air mass
    # mass_consvar[k, ij] = mass-weighted conservative variable
    radius, vcoord = model.planet.radius, model.vcoord
    consvar = gas(:p, :v).conservative_variable
    alloc() = similar(lon, nz, length(lon))
    mass_air, mass_consvar, ulon, ulat = alloc(), alloc(), alloc(), alloc()

    for k = 1:nz, ij in eachindex(lon)
        let lon = lon[ij], lat = lat[ij]
            ps, _ = case(lon, lat)
            p = pressure_level(2k - 1, ps, vcoord) # full level k
            _, uu, vv = case(lon, lat, p)
            ulon[k, ij], ulat[k, ij] = radius * uu, radius * vv
            p_lower = pressure_level(2k - 2, ps, vcoord) # lower interface
            p_upper = pressure_level(2k, ps, vcoord) # upper interface
            mg = p_lower - p_upper
            Phi_lower, _, _ = case(lon, lat, p_lower)
            Phi_upper, _, _ = case(lon, lat, p_upper)
            v = (Phi_upper - Phi_lower) / mg # dPhi = -v . dp
            mass_air[k, ij] = radius^2 * mg
            mass_consvar[k, ij] = (radius^2 * mg) * consvar(p, v)
        end
    end
    return mass_air, mass_consvar, ulon, ulat
end
