module Diagnostics

using CookBooks: CookBook
using CFDomains: primal_lonlat_from_cov!
using CFPlanets: lonlat_from_cov
using ManagedLoops: @with, @vec, @unroll

using ..Stencils: stencil, dot_product

function diagnostics()
    return CookBook(;
        ucov,
        mass_air,
        mass_consvar,
        ulonlat_i,
        ulon,
        ulat,
        pressure_i,
        pressure,
        surface_pressure_i,
        surface_pressure,
        consvar_i,
        consvar,
        temperature_i,
        temperature,
        geopotential_i,
        geopotential,
        gradp,
        ugradp_i,
        ugradp,
    )
end

# lon-lat diagnostics
ulon(interp, ulonlat_i) = interp(ulonlat_i.ulon)
ulat(interp, ulonlat_i) = interp(ulonlat_i.ulat)
pressure(interp, pressure_i) = interp(pressure_i)
surface_pressure(interp, surface_pressure_i) = interp(surface_pressure_i)
consvar(interp, consvar_i) = interp(consvar_i)
temperature(interp, temperature_i) = interp(temperature_i)
geopotential(interp, geopotential_i) = interp(geopotential_i)
ugradp(interp, ugradp_i) = interp(ugradp_i)

# diagnostics on native grid

ucov(state) = state.ucov
mass_air(model, state) = rescale_mass(model, state.mass_air)
mass_consvar(model, state) = rescale_mass(model, state.mass_consvar)
rescale_mass(model, mass) = (model.planet.radius^-2) * mass

surface_pressure_i(model, mass_air) =
    model.vcoord.ptop .+ reshape(sum(mass_air; dims = 1), :)

consvar_i(mass_air, mass_consvar) = @. mass_consvar / mass_air

temperature_i(model, pressure_i, consvar_i) =
    model.gas(:p, :consvar).temperature.(pressure_i, consvar_i)

function pressure_i(model, mass_air)
    # mass_air[ij, nz]
    p = similar(mass_air)
    @with model.mgr,
    let ijrange = 1:size(p, 2)
        ptop, nz = model.vcoord.ptop, size(p, 1)
        for ij in ijrange
            p[nz, ij] = ptop + mass_air[nz, ij] / 2
            for k = nz:-1:2
                p[k-1, ij] = p[k, ij] + (mass_air[k, ij] + mass_air[k-1, ij]) / 2
            end
        end
    end
    return p
end

function ulonlat_i(model, state)
    domain, planet, ucov = model.domain.layer, model.planet, state.ucov
    lon, lat = domain.lon_i, domain.lat_i
    coslon, sinlon = cos.(lon), sin.(lon)
    coslat, sinlat = cos.(lat), sin.(lat)
    ulon, ulat = similar(state.mass_air), similar(state.mass_air)
    args = domain.primal_deg,
    domain.primal_edge,
    domain.primal_perot_cov,
    coslon,
    sinlon,
    coslat,
    sinlat
    primal_lonlat_from_cov!(ulon, ulat, ucov, args...) do ij, ulon_ij, ulat_ij
        lonlat_from_cov(ulon_ij, ulat_ij, lon[ij], lat[ij], planet)
    end
    return (; ulon, ulat)
end

function geopotential_i(model, mass_air, mass_consvar, pressure_i)
    Phi = similar(pressure_i, size(pressure_i) .+ (1, 0))
    @with model.mgr, let ijrange = axes(pressure_i, 2)
        vol = model.gas(:p, :consvar).specific_volume
        for ij in ijrange
            Phi[1, ij] = model.Phis[ij]
            for k in axes(pressure_i, 1)
                consvar_ijk = mass_consvar[k, ij] / mass_air[k, ij]
                v = vol(pressure_i[k, ij], consvar_ijk)
                dPhi = mass_air[k, ij] * v
                Phi[k+1, ij] = Phi[k, ij] + dPhi
            end
        end
    end
    return Phi
end

function gradp(model, ucov, pressure_i) # 1-form, at edges
    gradp = similar(ucov)
    left_right = model.domain.layer.edge_left_right

    @with model.mgr, let (krange, ijrange) = axes(gradp)
        for ij in ijrange
            left, right = left_right[1, ij], left_right[2, ij]
            @vec for k in krange
                gradp[k, ij] = pressure_i[k, right] - pressure_i[k, left]
            end
        end
    end
    return gradp
end

function ugradp_i(model, pressure_i, ucov, gradp) # scalar, primal mesh
    ugradp = similar(pressure_i)
    domain, planet = model.domain.layer, model.planet

    @with model.mgr,
    let (krange, ijrange) = axes(ugradp)
        @inbounds for ij in ijrange
            deg = domain.primal_deg[ij]
            @unroll deg in 5:7 begin
                st = stencil(dot_product, Val(deg), ij, radius, domain)
                for k in krange
                    ugradp[k, ij] = dot_product(Val(deg), st, k, gradp, ucov)
                end
            end
        end
    end
    return ugradp
end

end # module
