module Voronoi

using CFDomains: VoronoiSphere, shell, VHLayout
using CFPlanets: ShallowTradPlanet, coriolis
using CFHydrostatics: initial_HPE_VH_collocated
import CFHydrostatics: HPE, initial_HPE_VH, HPE_diagnostics, HPE_tendencies!, HPE_remap!

function HPE(params, mgr, sph::VoronoiSphere, vcoord, geopotential, gas)
    (; radius, Omega) = params
    (; lon_i, lat_i, lon_v, lat_v, Av) = sph
    planet = ShallowTradPlanet(radius, Omega)
    fcov = Av.*coriolis.(Ref(planet), lon_v, lat_v)
    Phis = geopotential.(lon_i, lat_i)
    return HPE(mgr, vcoord, planet, shell(params.nz, sph), gas, fcov, Phis)
end

# called by initial_HPE
function initial_HPE_VH(model, nz, sph::VoronoiSphere, case)
    (; lon_i, lat_i, lon_e, lat_e, angle_e, de) = sph
    mass_air, mass_consvar, _, _ = initial_HPE_VH_collocated(model, nz, lon_i, lat_i, model.gas, case)
    _, _, ulon, ulat = initial_HPE_VH_collocated(model, nz, lon_e, lat_e, model.gas, case)

    ucov = similar(ulon)
    for k in 1:nz, ij in eachindex(de)
        sin_e, cos_e = sincos(angle_e[ij])
        ucov[k, ij] = de[ij]*(cos_e*ulon[k,ij] + sin_e*ulat[ij])
    end

    return (; mass_air, mass_consvar, ucov)
end

include("voronoi_dynamics.jl")
include("voronoi_diagnostics.jl")

HPE_diagnostics(_, ::VoronoiSphere) = Diagnostics.diagnostics()
HPE_tendencies!(dstate, scratch, model, ::VoronoiSphere, state, t) =
    Dynamics.tendencies_HV!(dstate, scratch, model, state, t)

end # module
