module Dynamics

using CFPlanets: lonlat_from_cov
using ManagedLoops: @with, @vec, @unroll
using MutatingOrNot: Void, void
using ..Stencils:
    centered_flux,
    dot_product,
    gradient,
    divergence,
    curl,
    TRiSK,
    average_ie,
    average_iv,
    average_ve

similar!(x, _...) = x
similar!(::Void, y...) = similar(y...)

function tendencies!(dstate, scratch, model, state, t)
    (; mass_air, mass_consvar, ucov) = state
    dmass_air, dmass_consvar, ducov = (dstate.mass_air, dstate.mass_consvar, dstate.ucov)

    # flux-form mass budgets
    (; flux_air, flux_consvar, consvar) = scratch.mass_budget
    dmass_air, dmass_consvar, flux_air, flux_consvar, consvar = mass_budget!(
        (dmass_air, dmass_consvar, flux_air, flux_consvar, consvar),
        model,
        (mass_air, mass_consvar, ucov),
    )

    # hydrostatic balance and Bernoulli function
    (; pressure, Phi, B, exner) = scratch.Bernoulli
    Phi, pressure = hydrostatic_balance!(Phi, pressure, model, mass_air, consvar)
    B, exner = Bernoulli!(B, exner, model, ucov, consvar, pressure, Phi)

    # Potential vorticity
    (; PV_e, PV_v) = scratch.PV
    PV_e, PV_v = potential_vorticity!(PV_e, PV_v, model, ucov, mass_air)

    # curl-form momentum budget
    ducov = curl_form!(ducov, model, PV_e, flux_air, B, consvar, exner)

    scratch = (
        mass_budget = (; flux_air, flux_consvar, consvar),
        Bernoulli = (; pressure, Phi, B, exner),
        PV = (; PV_e, PV_v),
    )

    return (mass_air = dmass_air, mass_consvar = dmass_consvar, ucov = ducov), scratch
end

function tendencies_HV!(dstate, scratch, model, state, t)
    (; mass_air, mass_consvar, ucov) = state
    dmass_air, dmass_consvar, ducov = (dstate.mass_air, dstate.mass_consvar, dstate.ucov)

    # flux-form mass budgets
    (; flux_air, flux_consvar, consvar) = scratch.mass_budget
    dmass_air, dmass_consvar, flux_air, flux_consvar, consvar = mass_budget!(
        (dmass_air, dmass_consvar, flux_air, flux_consvar, consvar),
        model,
        (mass_air, mass_consvar, ucov),
    )

    # hydrostatic balance and Bernoulli function
    # we switch to/from [ij,k] to make recurrences faster on GPU
    (; consvar_HV, mass_air_HV, pressure_HV, Phi_HV) = scratch.HV
    consvar_HV = transpose!(consvar_HV, consvar)
    mass_air_HV = transpose!(mass_air_HV, mass_air)
    Phi_HV, pressure_HV =
        hydrostatic_balance_HV!(Phi_HV, pressure_HV, model, mass_air_HV, consvar_HV)

    (; pressure, Phi, B, exner) = scratch.Bernoulli
    pressure = transpose!(pressure, pressure_HV)
    Phi = transpose!(Phi, Phi_HV)
    B, exner = Bernoulli!(B, exner, model, ucov, consvar, pressure, Phi)

    # Potential vorticity
    (; PV_e, PV_v) = scratch.PV
    PV_e, PV_v = potential_vorticity!(PV_e, PV_v, model, ucov, mass_air)

    # curl-form momentum budget
    ducov = curl_form!(ducov, model, PV_e, flux_air, B, consvar, exner)

    scratch = (
        mass_budget = (; flux_air, flux_consvar, consvar),
        HV = (; consvar_HV, mass_air_HV, pressure_HV, Phi_HV),
        Bernoulli = (; pressure, Phi, B, exner),
        PV = (; PV_e, PV_v),
    )

    return (mass_air = dmass_air, mass_consvar = dmass_consvar, ucov = ducov), scratch
end

transpose!(::Void, y) = permutedims(y, (2, 1))
transpose!(x, y) = permutedims!(x, y, (2, 1))

function mass_budget!(
    (dmass_air_, dmass_consvar_, flux_air_, flux_consvar_, consvar_),
    model,
    (mass_air, mass_consvar, ucov),
)
    consvar = similar!(consvar_, mass_air)
    dmass_air, dmass_consvar =
        similar!(dmass_air_, mass_air), similar!(dmass_consvar_, mass_consvar)
    flux_air, flux_consvar = similar!(flux_air_, ucov), similar!(flux_consvar_, ucov)

    domain, inv_rad2 = model.domain.layer, model.planet.radius^-2

    @with model.mgr, let (krange, ijrange) = axes(consvar)
        @inbounds for ij in ijrange
            @vec for k in krange
                consvar[k, ij] = mass_consvar[k, ij] * inv(mass_air[k, ij])
            end
        end
    end

    @with model.mgr, let (krange, ijrange) = axes(flux_air)
        @inbounds for ij in ijrange
            flux = centered_flux(domain, ij)
            avg = average_ie(domain, ij)
            @vec for k in krange
                flux_air[k, ij] = inv_rad2 * flux(mass_air, ucov, k)
                flux_consvar[k, ij] = flux_air[k, ij] * avg(consvar, k)
            end
        end
    end

    @with model.mgr, let (krange, ijrange) = axes(dmass_air)
        @inbounds for ij in ijrange
            deg = domain.primal_deg[ij]
            @unroll deg in 5:7 begin
                dvg = divergence(domain, ij, Val(deg))
                @vec for k in krange
                    dmass_air[k, ij] = -dvg(flux_air, k)
                    dmass_consvar[k, ij] = -dvg(flux_consvar, k)
                end
            end
        end
    end

    return dmass_air, dmass_consvar, flux_air, flux_consvar, consvar
end

function hydrostatic_balance!(Phi_, p_, model, mass_air, consvar)
    p = similar!(p_, mass_air)
    Phi = similar!(Phi_, p)

    ptop, nz = model.vcoord.ptop, size(p, 1)
    Phis = model.Phis # surface geopotential
    inv_rad2 = (model.planet.radius^-2) / 2
    vol = model.gas(:p, :consvar).specific_volume

    @with model.mgr, let ijrange = 1:size(p, 2)
        @inbounds for ij in ijrange
            p_top = ptop
            for k = nz:-1:1
                p_bot = p_top + inv_rad2 * mass_air[k, ij]
                p[k, ij] = (p_bot + p_top) / 2
                p_top = p_bot
            end
            Phi_bot = Phis[ij]
            for k in axes(Phi, 1)
                v = vol(p[k, ij], consvar[k, ij])
                Phi_top = Phi_bot + inv_rad2 * mass_air[k, ij] * v
                Phi[k, ij] = (Phi_top + Phi_bot) / 2
                Phi_bot = Phi_top
            end
        end
    end
    return Phi, p
end

function hydrostatic_balance_HV!(Phi_, p_, model, mass_air, consvar)
    p = similar!(p_, mass_air)
    Phi = similar!(Phi_, p)

    ptop, nz = model.vcoord.ptop, size(p, 2)
    Phis = model.Phis # surface geopotential
    inv_rad2 = (model.planet.radius^-2) / 2
    vol = model.gas(:p, :consvar).specific_volume

    @with model.mgr, let ijrange = axes(p, 1)
        @inbounds begin
            @vec for ij in ijrange
                p_top = ptop
                for k = nz:-1:1
                    p_bot = p_top + inv_rad2 * mass_air[ij, k]
                    p[ij, k] = (p_bot + p_top) / 2
                    p_top = p_bot
                end
                Phi_bot = Phis[ij]
                for k in axes(Phi, 2)
                    v = vol(p[ij, k], consvar[ij, k])
                    Phi_top = Phi_bot + inv_rad2 * mass_air[ij, k] * v
                    Phi[ij, k] = (Phi_top + Phi_bot) / 2
                    Phi_bot = Phi_top
                end
            end
        end
    end
    return Phi, p
end

function Bernoulli!(B_, exner_, model, ucov, consvar, p, Phi)
    B = similar!(B_, consvar)
    exner = similar!(exner_, consvar)

    radius = model.planet.radius
    Exner = model.gas(:p, :consvar).exner_functions
    domain = model.domain.layer
    degree = domain.primal_deg

    @with model.mgr,
    let (krange, ijrange) = axes(B)
        @inbounds for ij in ijrange
            deg = degree[ij]
            @unroll deg in 5:7 begin
                dot_ij = dot_product(domain, ij, Val(deg), radius)
                @vec for k in krange
                    consvar_ijk = consvar[k, ij]
                    h, v, exner_ijk = Exner(p[k, ij], consvar_ijk)
                    ke = dot_ij(ucov, ucov, k) / 2
                    exner[k, ij] = exner_ijk
                    B[k, ij] = Phi[k, ij] + ke + (h - consvar_ijk * exner_ijk)
                end
            end
        end
    end

    return B, exner
end

function potential_vorticity!(PV_e_, PV_v_, model, ucov, mass_air)
    nz = size(ucov, 1)
    fcov = model.fcov
    PV_v = similar!(PV_v_, fcov, nz, length(fcov))
    PV_e = similar!(PV_e_, ucov)
    domain = model.domain.layer

    @with model.mgr, let (krange, ijrange) = axes(PV_v)
        @inbounds for ij in ijrange
            curl_ij = curl(domain, ij)
            avg_ij = average_iv(domain, ij) # area-weighted sum from cells to vertices
            f_ij = fcov[ij] # = Coriolis * cell area Av
            @vec for k in krange
                zeta = curl_ij(ucov, k)
                m = avg_ij(mass_air, k)
                PV_v[k, ij] = (zeta + f_ij) / m
            end
        end
    end
    @with model.mgr, let (krange, ijrange) = axes(PV_e)
        @inbounds for ij in ijrange
            avg_ij = average_ve(domain, ij) # centered averaging from vertices to edges
            @vec for k in krange
                PV_e[k, ij] = avg_ij(PV_v, k)
            end
        end
    end
    return PV_e, PV_v
end

function curl_form!(ducov_, model, PV_e, flux_air, B, consvar, exner)
    ducov = similar!(ducov_, flux_air)
    domain = model.domain.layer

    #    @info "curl_form!" size(ducov) size(flux_air) size(PV_e)

    @with model.mgr,
    let (krange, ijrange) = axes(ducov)
        @inbounds for ij in ijrange
            grad_ij = gradient(domain, ij) # covariant gradient
            avg_ij = average_ie(domain, ij) # centered average from cells to edges
            @vec for k in krange
                ducov[k, ij] = -(grad_ij(B, k) + avg_ij(consvar, k) * grad_ij(exner, k))
            end
            for edge = 1:domain.trisk_deg[ij]
                # contribution of nearby `edge` to TRiSK operator at `ij`
                trisk_ij = TRiSK(domain, ij, edge)
                @vec for k in krange
                    ducov[k, ij] = trisk_ij(ducov, flux_air, PV_e, k)
                end
            end

        end
    end
    return ducov
end

end # module
