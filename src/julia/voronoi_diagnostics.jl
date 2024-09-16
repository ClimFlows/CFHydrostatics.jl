module Diagnostics

using CookBooks: CookBook
using CFDomains: primal_lonlat_from_cov!
using CFPlanets: lonlat_from_cov
using ManagedLoops: @with, @vec, @unroll

using ..Stencils: dot_product, centered_flux, divergence

diagnostics() = CookBook(;
    ulon,
    ulat,
    pressure,
    consvar,
    surface_pressure,
    temperature,
    geopotential,
    ugradp,
    dmass_air,
    Omega,
    ucov_e,
    mass_air_i,
    mass_consvar_i,
    ulonlat_i,
    pressure_i,
    surface_pressure_i,
    consvar_i,
    temperature_i,
    geopotential_i,
    gradp_e,
    ugradp_i,
    dmass_air_i,
    Omega_i,
)

# lon-lat diagnostics
ulon(to_lonlat, ulonlat_i) = to_lonlat(ulonlat_i.ulon)
ulat(to_lonlat, ulonlat_i) = to_lonlat(ulonlat_i.ulat)
pressure(to_lonlat, pressure_i) = to_lonlat(pressure_i)
surface_pressure(to_lonlat, surface_pressure_i) = to_lonlat(surface_pressure_i)
consvar(to_lonlat, consvar_i) = to_lonlat(consvar_i)
temperature(to_lonlat, temperature_i) = to_lonlat(temperature_i)
geopotential(to_lonlat, geopotential_i) = to_lonlat(geopotential_i)
ugradp(to_lonlat, ugradp_i) = to_lonlat(ugradp_i)
dmass_air(to_lonlat, dmass_air_i) = to_lonlat(dmass_air_i)
Omega(to_lonlat, Omega_i) = to_lonlat(Omega_i)

# diagnostics on native grid

ucov_e(state) = state.ucov
mass_air_i(model, state) = rescale_mass(model, state.mass_air)
mass_consvar_i(model, state) = rescale_mass(model, state.mass_consvar)
rescale_mass(model, mass) = (model.planet.radius^-2) * mass

surface_pressure_i(model, mass_air_i) =
    model.vcoord.ptop .+ reshape(sum(mass_air_i; dims = 1), :)

consvar_i(mass_air_i, mass_consvar_i) = @. mass_consvar_i / mass_air_i

temperature_i(model, pressure_i, consvar_i) =
    model.gas(:p, :consvar).temperature.(pressure_i, consvar_i)

dmass_air_i(model, mass_air_i, ucov_e) = dmass_i(model, mass_air_i, ucov_e)
dmass_consvar_i(model, mass_consvar_i, ucov_e) = dmass_i(model, mass_consvar_i, ucov_e)

function pressure_i(model, mass_air_i)
    # mass_air_i[ij, nz]
    p = similar(mass_air_i)
    @with model.mgr,
    let ijrange = 1:size(p, 2)
        ptop, nz = model.vcoord.ptop, size(p, 1)
        for ij in ijrange
            p[nz, ij] = ptop + mass_air_i[nz, ij] / 2
            for k = nz:-1:2
                p[k-1, ij] = p[k, ij] + (mass_air_i[k, ij] + mass_air_i[k-1, ij]) / 2
            end
        end
    end
    return p
end

function ulonlat_i(model, state)
    domain, planet, ucov_e = model.domain.layer, model.planet, state.ucov
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
    primal_lonlat_from_cov!(ulon, ulat, ucov_e, args...) do ij, ulon_ij, ulat_ij
        lonlat_from_cov(ulon_ij, ulat_ij, lon[ij], lat[ij], planet)
    end
    return (; ulon, ulat)
end

function geopotential_i(model, mass_air_i, mass_consvar_i, pressure_i)
    Phi = similar(pressure_i, size(pressure_i) .+ (1, 0))
    @with model.mgr, let ijrange = axes(pressure_i, 2)
        vol = model.gas(:p, :consvar).specific_volume
        for ij in ijrange
            Phi[1, ij] = model.Phis[ij]
            for k in axes(pressure_i, 1)
                consvar_ijk = mass_consvar_i[k, ij] / mass_air_i[k, ij]
                v = vol(pressure_i[k, ij], consvar_ijk)
                dPhi = mass_air_i[k, ij] * v
                Phi[k+1, ij] = Phi[k, ij] + dPhi
            end
        end
    end
    return Phi
end

function gradp_e(model, ucov_e, pressure_i) # 1-form, at edges
    gradp = similar(ucov_e)
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

function ugradp_i(model, pressure_i, ucov_e, gradp_e) # scalar, primal mesh
    ugradp = similar(pressure_i)
    domain, radius = model.domain.layer, model.planet.radius

    @with model.mgr,
    let (krange, ijrange) = axes(ugradp)
        @inbounds for ij in ijrange
            deg = domain.primal_deg[ij]
            @unroll deg in 5:7 begin
                st = dot_product(Val(deg), ij, domain, radius)
                for k in krange
                    ugradp[k, ij] = dot_product(Val(deg), st, k, gradp_e, ucov_e)
                end
            end
        end
    end
    return ugradp
end

function dmass_i(model, mass_i, ucov_e) # scalar, primal mesh
    flux, dmass = similar(ucov_e), similar(mass_i)
    domain, inv_rad2 = model.domain.layer, model.planet.radius^-2

    @with model.mgr,
    let (krange, ijrange) = axes(flux)
        @inbounds for ij in ijrange
            st = centered_flux(ij, domain)
            for k in krange
                flux[k, ij] = inv_rad2 * centered_flux(st, k, mass_i, ucov_e)
            end
        end
    end

    @with model.mgr,
    let (krange, ijrange) = axes(dmass)
        @inbounds for ij in ijrange
            deg = domain.primal_deg[ij]
            @unroll deg in 5:7 begin
                st = divergence(Val(deg), ij, domain)
                for k in krange
                    dmass[k, ij] = -divergence(Val(deg), st, k, flux)
                end
            end
        end
    end

    return dmass
end

function Omega_i(model, ugradp_i, dmass_air_i)    #= pressure_i, ugradPhi_i, mass_air_i, mass_consvar_i, dmass_consvar_i =#

    # dmass is a scalar (O-form in kg/m²/s)
    # consvar is a scalar (O-form in kg/m²/s)
    Omega, Phi_dot = similar(ugradp_i), similar(ugradp_i)
    dp_mid = Phi_dot # use Phi_dot as buffer for dp_mid

    #    @with model.mgr,
    let ijrange = axes(ugradp_i, 2)
        nz = size(ugradp_i, 1)
        volume = model.gas(:p, :consvar).volume_functions

        for ij in ijrange
            # top_down: dp, Omega
            dp_top = zero(Omega[1, ij])
            for k = nz:-1:1
                dp_bot = dp_top + dmass_air_i[k, ij]
                dp_mid[k, ij] = (dp_top + dp_bot) / 2
                Omega[k, ij] = dp_mid[k, ij] + ugradp_i[k, ij]
                dp_top = dp_bot
            end
            #=
            # bottom-up: Phi_dot
            dPhi = zero(Phi_dot[1, ij])
            for k = 1:nz
                consvar = mass_consvar_i[k, ij] / mass_air_i[k, ij]
                mass_dconsvar = dmass_consvar_i[k, ij] - consvar * dmass_air_i[k, ij]
                v, dv_dp, dv_dconsvar = volume(pressure_i[k, ij], consvar)
                ddPhi =
                    v * dmass_air[k, ij] +
                    dv_dconsvar * mass_dconsvar +
                    dv_dp * mass_air[k, ij] * dp_mid[k, ij]
                # here we take care to write into Phi_dot *after* reading from dp_mid
                Phi_dot[k, ij] = (dPhi + ddPhi / 2) + ugradPhi_i[k, ij]
                dPhi += ddPhi
            end
            =#
        end

    end
    return Omega
end

end # module
