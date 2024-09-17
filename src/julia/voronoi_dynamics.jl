module Dynamics

using CFPlanets: lonlat_from_cov
using ManagedLoops: @with, @vec, @unroll
using MutatingOrNot: Void, void
using ..Stencils: centered_flux, divergence, dot_product

similar!(x, _...) = x
similar!(::Void, y...) = similar(y...)

function tendencies!(dstate, scratch, model, state, t)
    (; mass_air, mass_consvar, ucov) = state
    dmass_air, dmass_consvar, ducov = (dstate.mass_air, dstate.mass_consvar, dstate.ducov)

    # flux-form mass budgets
    (; flux_air, flux_consvar, consvar) = scratch.mass_budget
    dmass_air, dmass_consvar, flux_air, flux_consvar = mass_budget!(
        (dmass_air, dmass_consvar, flux_air, flux_consvar, consvar),
        model,
        (mass_air, mass_consvar, ucov),
    )

    # hydrostatic balance and Bernoulli function
    (; pressure, Phi, B, exner) = scratch.Bernoulli
    pressure, Phi = hydrostatic_balance!(pressure, Phi, model, mass_air, consvar)
    B, exner = Bernoulli!(B, exner, model, ucov, consvar, pressure, Phi)

    # Potential vorticity
    (; PV_e, PV_v) = scratch.PV
    PV_e, PV_v = potential_vorticity!(PV_v, PV_e, model, ucov, mass_air)

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
    dmass_air, dmass_consvar, ducov = (dstate.mass_air, dstate.mass_consvar, dstate.ducov)

    # flux-form mass budgets
    (; flux_air, flux_consvar, consvar) = scratch.mass_budget
    dmass_air, dmass_consvar, flux_air, flux_consvar = mass_budget!(
        (dmass_air, dmass_consvar, flux_air, flux_consvar, consvar),
        model,
        (mass_air, mass_consvar, ucov),
    )

    # hydrostatic balance and Bernoulli function
    # we switch to/from [ij,k] to make recurrences faster on GPU
    (; consvar_HV, mass_air_HV, pressure_HV, Phi_HV) = scratch.HV
    consvar_HV = transpose!(consvar_HV, consvar)
    mass_air_HV = transpose!(mass_air_HV, mass_air)
    pressure_HV, Phi_HV =
        hydrostatic_balance_HV!(pressure_HV, Phi_HV, model, mass_air_HV, consvar_HV)

    (; pressure, Phi, B, exner) = scratch.Bernoulli
    pressure = transpose!(pressure, pressure_HV)
    Phi = transpose!(Phi, Phi_HV)
    B, exner = Bernoulli!(B, exner, model, ucov, consvar, pressure, Phi)

    # Potential vorticity
    (; PV_e, PV_v) = scratch.PV
    PV_e, PV_v = potential_vorticity!(scratch.PV_v, scratch.PV_i, model, ucov, mass_air)

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

transpose!(::Void, y) = permutedims(y, (1, 2))
transpose!(x, y) = permutedims!(x, y, (1, 2))

function mass_budget!(
    (dmass_air, dmass_consvar, flux_air, flux_consvar, consvar),
    model,
    (mass_air, mass_consvar, ucov),
)
    consvar = similar!(consvar, mass_air)
    dmass_air, dmass_consvar =
        similar!(dmass_air, mass_air), similar!(dmass_consvar, mass_consvar)
    flux_air, flux_consvar = similar!(flux_air, ucov), similar!(flux_consvar, ucov)

    domain, inv_rad2 = model.domain.layer, model.planet.radius^-2

    @with model.mgr, let (krange, ijrange) = axes(consvar)
        for ij in ijrange
            @vec for k in krange
                consvar[k, ij] = mass_consvar[k, ij] * inv(mass_air[k, ij])
            end
        end
    end

    @with model.mgr,
    let (krange, ijrange) = axes(flux_air)
        for ij in ijrange
            st = centered_flux(ij, domain)
            @vec for k in krange
                flux_air[k, ij] = inv_rad2 * centered_flux(st, k, mass_i, ucov_e)
                flux_consvar[k, ij] = flux_air[k, ij] * centered_average(st, k, consvar)
            end
        end
    end

    @with model.mgr,
    let (krange, ijrange) = axes(dmass)
        for ij in ijrange
            deg = domain.primal_deg[ij]
            @unroll deg in 5:7 begin
                st = divergence(Val(deg), ij, domain)
                @vec for k in krange
                    dmass_air[k, ij] = -divergence(Val(deg), st, k, flux_air)
                    dmass_consvar[k, ij] = -divergence(Val(deg), st, k, flux_consvar)
                end
            end
        end
    end

    return dmass_air, dmass_consvar, flux_air, flux_consvar, consvar
end

function hydrostatic_balance_VH!(Phi, p, model, mass_air, consvar)
    p = similar!(p, mass_air)
    Phi = similar!(Phi, p)

    ptop, nz = model.vcoord.ptop, size(p, 1)
    Phis = model.Phis # surface geopotential
    inv_rad2 = (model.planet.radius^-2) / 2
    vol = model.gas(:p, :consvar).specific_volume

    @with model.mgr, let ijrange = 1:size(p, 2)
        for ij in ijrange
            p_top = ptop
            for k = nz:-1:2
                p_bot = p_top + inv_rad2 * mass_air[k, ij]
                p[nz, ij] = (p_bot + p_top) / 2
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

function Bernoulli!(B, exner, model, ucov, consvar, p, Phi)
    B = similar!(B, consvar)
    exner = similar!(exner, consvar)
    radius = model.planet.radius
    Exner = model.gas(:p, :consvar).exner_functions
    domain = model.domain.layer
    degree = domain.primal_deg

    @with model.mgr,
    let (krange, ijrange) = axes(B)
        for ij in ijrange
            deg = degree[ij]
            @unroll deg in 5:7 begin
                dp = dot_product(Val(deg), ij, domain, radius)
                @vec for k in krange
                    consvar_ijk = consvar[k, ij]
                    h, v, exner_ijk = Exner(p[k, ij], consvar_ijk)
                    ke = dot_product(Val(deg), dp, k, ucov, ucov) / 2
                    exner[k, ij] = exner_ijk
                    B[k, ij] = Phi[k, ij] + ke + (h - consvar_ijk * exner_ijk)
                end
            end
        end
    end
end

function potential_vorticity!(PV_v, PV_e, model, ucov, mass_air)
    nz = size(ucov, 1)
    fcov = model.fcov
    PV_v = similar!(PV_v, nz, length(fcov))
    PV_e = similar!(PV_e, ucov)
    domain = model.domain.layer

    @with model.mgr, let (krange, ijrange) = axes(PV_v)
        for ij in ijrange
            curl_ij = curl(ij, domain)
            avg_ij = average_iv(ij, domain) # area-weighted sum from cells to vertices
            f_ij = fcov[ij] # = Coriolis * cell area Av
            @vec for k in krange
                zeta = curl_ij(ucov, k)
                m = avg_ij(mass_air, k)
                PV_v[k, ij] = (zeta + f_ij) / m
            end
        end
    end
    @with model.mgr, let (krange, ijrange) = axes(PV_e)
        for ij in ijrange
            avg_ij = average_ve(ij, domain) # centered averaging from vertices to edges
            @vec for k in krange
                PV_e[k, ij] = avg_ij(PV_v, k)
            end
        end
    end
    return PV_v, PV_e
end

function curl_form!(ducov, model, PV_e, flux_air, B, consvar, exner)
    ducov = similar!(ducov, flux_air)
    domain = model.domain.layer

    @with model.mgr, let (krange, ijrange) = axes(ducov)
        for ij in ijrange
            trisk_ij = TRiSK(ij, domain) # TRiSK operator
            grad_ij = DEC_grad(ij, domain) # covariant gradient
            avg_ij = average_ie(ij, domain) # centered average from cells to edges
            @vec for k in krange
                ducov[k, ij] = -trisk_ij(PV_e, flux_air, k) - (grad_ij(B, k) + avg_ij(consvar, k)*grad_ij(exner, k) )
            end
        end
    end
    return ducov
end

end # module



