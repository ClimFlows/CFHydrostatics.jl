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
    (; Ai, lon_i, lat_i, lon_e, lat_e, angle_e, de) = sph
    masscov_air, masscov_consvar, _, _ = initial_HPE_VH_collocated(model, nz, lon_i, lat_i, model.gas, case)
    _, _, ulon, ulat = initial_HPE_VH_collocated(model, nz, lon_e, lat_e, model.gas, case)

    # prognostic variables are covariant
    # air mass and θ mass : value per unit area * cell area
    for k in 1:nz, ij in eachindex(Ai)
        masscov_air[k, ij] *= Ai[ij]
        masscov_consvar[k, ij] *= Ai[ij]
    end

    # momentum: component normal to Voronoi edge * triangular edge length
    ucov = similar(ulon)
    for k in 1:nz, ij in eachindex(de)
        sin_e, cos_e = sincos(angle_e[ij])
        ucov[k, ij] = de[ij]*(cos_e*ulon[k,ij] + sin_e*ulat[ij])
    end

    return (; masscov_air, masscov_consvar, ucov)
end

include("voronoi_dynamics.jl")
include("voronoi_diagnostics.jl")

HPE_diagnostics(_, ::VoronoiSphere) = Diagnostics.diagnostics()
HPE_tendencies!(dstate, scratch, model, ::VoronoiSphere, state, t) =
    Dynamics.tendencies_HV!(dstate, scratch, model, state, t)
HPE_tendencies!(slow, fast, scratch, model, ::VoronoiSphere, state, t, tau) =
    Dynamics.tendencies_HV!(slow, fast, scratch, model, state, t, tau)

end # module
