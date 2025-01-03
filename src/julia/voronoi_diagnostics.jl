module Diagnostics

using CookBooks: CookBook
using CFDomains: Stencils, primal_lonlat_from_cov!
using CFPlanets
using ManagedLoops: @with, @vec, @unroll
using MutatingOrNot: void
using ..Dynamics: tendencies_HV!

diagnostics() = CookBook(;
    ulon,
    ulat,
    pressure,
    consvar,
    surface_pressure,
    temperature,
    geopotential,
    Omega,
    Phi_dot,
    ucov_e,
    mass_air_i,
    mass_consvar_i,
    ulonlat_i,
    pressure_i,
    surface_pressure_i,
    consvar_i,
    temperature_i,
    geopotential_i,
    kinetic_energy_i,
    gradp_e,
    gradPhi_e,
    dmass_air_i,
    dmass_consvar_i,
    lonlat_from_cov,
    vertical_velocities,
    tendencies,
    dulonlat_i,
    dulon,
    dulat
)

# lon-lat diagnostics

ulon(to_lonlat, ulonlat_i) = to_lonlat(ulonlat_i.ulon)
ulat(to_lonlat, ulonlat_i) = to_lonlat(ulonlat_i.ulat)
pressure(to_lonlat, pressure_i) = to_lonlat(pressure_i)
surface_pressure(to_lonlat, surface_pressure_i) = to_lonlat(surface_pressure_i)
consvar(to_lonlat, consvar_i) = to_lonlat(consvar_i)
temperature(to_lonlat, temperature_i) = to_lonlat(temperature_i)
geopotential(to_lonlat, geopotential_i) = to_lonlat(geopotential_i)
Omega(to_lonlat, vertical_velocities) = to_lonlat(vertical_velocities.Omega)
Phi_dot(to_lonlat, vertical_velocities) = to_lonlat(vertical_velocities.Phi_dot)

# diagnostics on native grid

ucov_e(state) = state.ucov
mass_air_i(model, state) = rescale_mass(model, state.mass_air)
mass_consvar_i(model, state) = rescale_mass(model, state.mass_consvar)
rescale_mass(model, mass) = (model.planet.radius^-2) * mass

ulonlat_i(ucov_e, lonlat_from_cov) = lonlat_from_cov(ucov_e)

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

function lonlat_from_cov(model, state)
    domain, planet = model.domain.layer, model.planet
    lon, lat = domain.lon_i, domain.lat_i
    coslon, sinlon = cos.(lon), sin.(lon)
    coslat, sinlat = cos.(lat), sin.(lat)
    args = (
        domain.primal_deg,
        domain.primal_edge,
        domain.primal_perot_cov,
        coslon,
        sinlon,
        coslat,
        sinlat,
    )
    function work(ucov)
        ulon, ulat = similar(state.mass_air), similar(state.mass_air)
        primal_lonlat_from_cov!(ulon, ulat, ucov, args...) do ij, ulon_ij, ulat_ij
            CFPlanets.lonlat_from_cov(ulon_ij, ulat_ij, lon[ij], lat[ij], planet)
        end
        return (; ulon, ulat)
    end
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

function kinetic_energy_i(model, ucov_e, mass_air_i)
    ke = similar(mass_air_i)
    domain = model.domain.layer
    half_metric = (model.planet.radius^-2)/2

    @with model.mgr,
    let (krange, ijrange) = axes(ke)
        for ij in ijrange
            deg = domain.primal_deg[ij]
            @unroll deg in 5:7 begin
                dot_product = Stencils.dot_product(domain, ij, Val(deg))
                @vec for k in krange
                    ke[k, ij] = half_metric * dot_product(ucov_e, ucov_e, k)
                end
            end
        end
    end
    return ke
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

function gradPhi_e(model, ucov_e, geopotential_i) # 1-form, at edges
    gradPhi = similar(ucov_e)
    left_right = model.domain.layer.edge_left_right

    @with model.mgr,
    let (krange, ijrange) = axes(gradPhi)
        for ij in ijrange
            left, right = left_right[1, ij], left_right[2, ij]
            @vec for k in krange
                gradPhi[k, ij] =
                    (
                        (geopotential_i[k, right] + geopotential_i[k+1, right]) -
                        (geopotential_i[k, left] + geopotential_i[k+1, left])
                    ) / 2
            end
        end
    end
    return gradPhi
end

function dmass_i(model, mass_i, ucov_e) # scalar, primal mesh
    flux, dmass = similar(ucov_e), similar(mass_i)
    domain, inv_rad2 = model.domain.layer, model.planet.radius^-2

    @with model.mgr, let (krange, ijrange) = axes(flux)
        @inbounds for ij in ijrange
            flux_ij = Stencils.centered_flux(domain, ij)
            for k in krange
                flux[k, ij] = inv_rad2 * flux_ij(mass_i, ucov_e, k)
            end
        end
    end

    @with model.mgr, let (krange, ijrange) = axes(dmass)
        @inbounds for ij in ijrange
            deg = domain.primal_deg[ij]
            @unroll deg in 5:7 begin
                dvg = Stencils.divergence(domain, ij, Val(deg))
                for k in krange
                    dmass[k, ij] = -dvg(flux, k)
                end
            end
        end
    end

    return dmass
end

function vertical_velocities(
    model,
    ucov_e,
    gradp_e,
    gradPhi_e,
    dmass_air_i,
    dmass_consvar_i,
    pressure_i,
    mass_air_i,
    mass_consvar_i,
)
    # dmass is a scalar (O-form in kg/m²/s)
    # consvar is a scalar (O-form in kg/m²/s)
    Omega, Phi_dot = similar(mass_air_i), similar(mass_air_i)
    dp_mid = Phi_dot # use Phi_dot as buffer for dp_mid

    volume = model.gas(:p, :consvar).volume_functions
    domain = model.domain.layer
    metric = model.planet.radius^-2 # contravariant metric tensor (diagonal)

    @with model.mgr,
    let ijrange = axes(Omega, 2)
        nz = size(Omega, 1)
        for ij in ijrange
            deg = domain.primal_deg[ij]
            @unroll deg in 5:7 begin
                dot_product = Stencils.dot_product(domain, ij, Val(deg))
                # top_down: dp, Omega
                dp_top = zero(Omega[1, ij])
                for k = nz:-1:1
                    dp_bot = dp_top + dmass_air_i[k, ij]
                    dp_mid[k, ij] = (dp_top + dp_bot) / 2
                    ugradp_ijk = metric * dot_product(ucov_e, gradp_e, k)
                    Omega[k, ij] = dp_mid[k, ij] + ugradp_ijk
                    dp_top = dp_bot
                end
                # bottom-up: Phi_dot
                dPhi = zero(Phi_dot[1, ij])
                for k = 1:nz
                    consvar = mass_consvar_i[k, ij] / mass_air_i[k, ij]
                    mass_dconsvar = dmass_consvar_i[k, ij] - consvar * dmass_air_i[k, ij]
                    v, dv_dp, dv_dconsvar = volume(pressure_i[k, ij], consvar)
                    ddPhi =
                        v * dmass_air_i[k, ij] +
                        dv_dconsvar * mass_dconsvar +
                        dv_dp * mass_air_i[k, ij] * dp_mid[k, ij]
                    # here we take care to write into Phi_dot *after* reading from dp_mid
                    ugradPhi_ijk = metric * dot_product(ucov_e, gradPhi_e, k)
                    Phi_dot[k, ij] = (dPhi + ddPhi / 2) + ugradPhi_ijk
                    dPhi += ddPhi
                end
            end
        end
    end
    return (; Omega, Phi_dot)
end

#================= tendencies ===================#

tendencies(model, state) =
    tendencies_HV!(void, void, model, state, nothing)

dulonlat_i(tendencies, lonlat_from_cov) = lonlat_from_cov(tendencies[1].ucov)
dulon(dulonlat_i, to_lonlat) = to_lonlat(dulonlat_i.ulon)
dulat(dulonlat_i, to_lonlat) = to_lonlat(dulonlat_i.ulat)

end # module
