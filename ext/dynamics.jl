module Dynamics

using MutatingOrNot: void, Void

using ManagedLoops: @with, @vec, no_simd
using SHTnsSpheres:
    analysis_scalar!,
    synthesis_scalar!,
    analysis_vector!,
    synthesis_vector!,
    synthesis_spheroidal!,
    divergence!,
    curl!,
    shtns_alloc,
    erase,
    batch

using CFHydrostatics: debug_flags

# similar!(x,y) allocates only if x::Void
similar!(::Void, y) = similar(y)
similar!(x, y) = x

include("fused.jl")
include("zero_array.jl")

# spectral fields are suffixed with _spec
# vector, spectral = (spheroidal, toroidal)
# vector, spatial = (ucolat, ulon)

vector_spec(spheroidal, toroidal) = (; spheroidal, toroidal)
vector_spat(ucolat, ulon) = (; ucolat, ulon)
HPE_state(mass_air_spec, mass_consvar_spec, uv_spec) =
    (; mass_air_spec, mass_consvar_spec, uv_spec)

function tendencies!(dstate, scratch, model, state, t)
    (; locals, locals_dmass, locals_duv) = scratch
    (; uv, mass_air, mass_consvar, p, B, exner, consvar, geopot) = locals
    (; mass_air_spec, mass_consvar_spec, uv_spec) = state
    dmass_air_spec, dmass_consvar_spec, duv_spec = dstate

    sph, metric, fcov = model.domain.layer, model.planet.radius^-2, model.fcov

#    fused_mass_budgets! = Fused(mass_budgets!)
#    fused_curl_form! = Fused(curl_form!)
    fused_mass_budgets! = mass_budgets!
    fused_curl_form! = curl_form!

    # flux-form mass balance
    (; dmass_air_spec, dmass_consvar_spec, uv, mass_air, mass_consvar), locals_dmass =
        fused_mass_budgets!(
            (; dmass_air_spec, dmass_consvar_spec, uv, mass_air, mass_consvar),
            scratch.locals_dmass,
            (; mass_air_spec, mass_consvar_spec, uv_spec),
            sph,
            sph.laplace,
            metric,
        )

    # hydrostatic balance, geopotential, exner function
    p = hydrostatic_pressure!(p, model, mass_air)
    B, exner, consvar, geopot =
        Bernoulli!((B, exner, consvar, geopot), (mass_air, mass_consvar, p, uv), model)

    # curl-form momentum balance
    (; duv_spec), locals_duv = fused_curl_form!(
        (; duv_spec),
        scratch.locals_duv,
        (; exner, consvar, B, uv_spec, uv),
        sph,
        sph.laplace,
        metric,
        fcov,
    )

    locals = (; uv, mass_air, mass_consvar, p, B, exner, consvar, geopot)
    scratch = (; locals, locals_duv, locals_dmass)

    return HPE_state(dmass_air_spec, dmass_consvar_spec, duv_spec), scratch
end

function tendencies!(slow, fast, scratch, model, state, t, tau)
    (; locals, locals_duv_fast, locals_slow) = scratch
    (; uv, mass_air, mass_consvar, p, B, exner, consvar, geopot) = locals
    (; mass_air_spec, mass_consvar_spec, uv_spec) = state
    dmass_air_spec_slow, dmass_consvar_spec_slow, duv_spec_slow = slow
    dmass_air_spec_fast, dmass_consvar_spec_fast, duv_spec_fast = fast

    sph, metric, fcov = model.domain.layer, model.planet.radius^-2, model.fcov

    # compute fast tendencies for ucov and update ucov
    mass_air = synthesis_scalar!(mass_air, mass_air_spec, sph)
    mass_consvar = synthesis_scalar!(mass_consvar, mass_consvar_spec, sph)
    p = hydrostatic_pressure!(p, model, mass_air)

    B, exner, consvar, geopot =
        Bernoulli_fast!((B, exner, consvar, geopot), (mass_air, mass_consvar, p), model)

    duv_spec_fast, locals_duv_fast = curl_form_fast!(
        duv_spec_fast,
        scratch.locals_duv_fast,
        (; exner, consvar, B),
        sph )
    
    # flux-form mass balance and curl-form momentum balance for slow terms
    (; dmass_air_spec_slow, dmass_consvar_spec_slow, duv_spec_slow), locals_slow =
        Fused(tendencies_slow!)(
            (; dmass_air_spec_slow, dmass_consvar_spec_slow, duv_spec_slow),
            scratch.locals_slow,
            (; mass_air, mass_consvar, uv_spec, duv_spec_fast),
            sph,
            sph.laplace,
            metric, fcov, tau,
        )

    locals = (; uv, mass_air, mass_consvar, p, B, exner, consvar, geopot)
    scratch = (; locals, locals_duv_fast, locals_slow)

    dmass_air_spec_fast = ZeroArray(axes(dmass_air_spec_slow))
    dmass_consvar_spec_fast = ZeroArray(axes(dmass_consvar_spec_slow))

    slow = HPE_state(dmass_air_spec_slow, dmass_consvar_spec_slow, duv_spec_slow)
    fast = HPE_state(dmass_air_spec_fast, dmass_consvar_spec_fast, duv_spec_fast)
    return slow, fast, scratch
end

#=========================== flux-form mass budget ======================
       ∂Φ/∂t = -∇(Φu, Φv)
     uv is the momentum 1-form = a*(u,v)
     gh is the 2-form a²Φ
     divergence! is relative to the unit sphere
       => scale flux by contravariant metric factor radius^-2
========================================================================#

function mass_budgets!(outputs, locals, inputs, sph, laplace, factor)
    (; mass_air_spec, mass_consvar_spec, uv_spec) = inputs
    (; dmass_air_spec, dmass_consvar_spec, uv, mass_air, mass_consvar) = outputs
    (; fx, fy, flux_air_spec, fcx, fcy, flux_consvar_spec) = locals

    (ux, uy) = uv = synthesis_vector!(uv, uv_spec, sph)

    mass_air = synthesis_scalar!(mass_air, mass_air_spec, sph)
    fx = @. fx = -factor * ux * mass_air
    fy = @. fy = -factor * uy * mass_air
    flux_air_spec = analysis_vector!(flux_air_spec, erase(vector_spat(fx, fy)), sph)
    dmass_air_spec = @. dmass_air_spec = flux_air_spec.spheroidal * laplace

    mass_consvar = synthesis_scalar!(mass_consvar, mass_consvar_spec, sph)
    fcx = @. fcx = -factor * ux * mass_consvar
    fcy = @. fcy = -factor * uy * mass_consvar
    flux_consvar_spec =
        analysis_vector!(flux_consvar_spec, erase(vector_spat(fcx, fcy)), sph)
    dmass_consvar_spec = @. dmass_consvar_spec = flux_consvar_spec.spheroidal * laplace

    return (; dmass_air_spec, dmass_consvar_spec, uv, mass_air, mass_consvar),
    (; fx, fy, flux_air_spec, fcx, fcy, flux_consvar_spec)
end

#====================== curl-form momentum budget ================
       ∂u/∂t = (f+ζ)v - θ∂π/∂x- ∂B/∂x
       ∂v/∂t = -(f+ζ)u - θ∂π/∂y - ∂B/∂y
       B = (u²+v²)2 + gz + (h - θπ)
     uv is momentum = a*(u,v)
     curl! is relative to the unit sphere
     fcov, zeta and gh are the 2-forms a²f, a²ζ, a²Φ
       => scale B and qflux by contravariant metric factor radius^-2
==================================================================#

function curl_form!(outputs, locals, inputs, sph, laplace, factor, fcov)
    (; exner, consvar, B, uv_spec, uv) = inputs
    (; duv_spec) = outputs
    (; qflux_spec, B_spec, fx, fy, zeta, zeta_spec, grad_exner, exner_spec) = locals

    flags = debug_flags()

    exner_spec = analysis_scalar!(exner_spec, erase(exner), sph)
    (gradx, grady) = grad_exner = synthesis_spheroidal!(grad_exner, exner_spec, sph)
    zeta_spec = @. zeta_spec = -laplace * uv_spec.toroidal # curl
    zeta = synthesis_scalar!(zeta, zeta_spec, sph)
    (ux, uy) = uv
    fx = @. fx =  (flags.qU) * factor * (zeta + fcov) * uy - (flags.CgradExner)*consvar * gradx
    fy = @. fy = -(flags.qU) * factor * (zeta + fcov) * ux - (flags.CgradExner)*consvar * grady

    qflux_spec = analysis_vector!(qflux_spec, erase(vector_spat(fx, fy)), sph)

    B_spec = analysis_scalar!(B_spec, erase(B), sph)
    duv_spec = vector_spec(
        (@. duv_spec.spheroidal = qflux_spec.spheroidal - (flags.gradB)*B_spec),
        (@. duv_spec.toroidal = qflux_spec.toroidal),
    )
    return (; duv_spec),
    (; qflux_spec, B_spec, fx, fy, zeta, zeta_spec, grad_exner, exner_spec)
end

function curl_form_fast!(duv_spec_fast, locals, inputs, sph)
    (; exner, consvar, B) = inputs
    (; exner_spec, grad_exner, fx, fy, qflux_spec, B_spec) = locals

    flags = debug_flags()

    exner_spec = analysis_scalar!(exner_spec, erase(exner), sph)
    (gradx, grady) = grad_exner = synthesis_spheroidal!(grad_exner, exner_spec, sph)
    fx = @. fx = - (flags.CgradExner)*consvar * gradx
    fy = @. fy = - (flags.CgradExner)*consvar * grady
    qflux_spec = analysis_vector!(qflux_spec, erase(vector_spat(fx, fy)), sph)

    B_spec = analysis_scalar!(B_spec, erase(B), sph)
    duv_spec_fast = vector_spec(
        (@. duv_spec_fast.spheroidal = qflux_spec.spheroidal - (flags.gradB)*B_spec),
        (@. duv_spec_fast.toroidal = qflux_spec.toroidal),
    )
    return duv_spec_fast,
    (; exner_spec, grad_exner, fx, fy, qflux_spec, B_spec)
end

function tendencies_slow!(outputs, locals, inputs, sph, laplace, factor, fcov, tau)
    (; mass_air, mass_consvar, uv_spec, duv_spec_fast) = inputs
    (; dmass_air_spec_slow, dmass_consvar_spec_slow, duv_spec_slow) = outputs
    (; uv_spec_new, uv, fx, fy, flux_air_spec, fcx, fcy, flux_consvar_spec) = locals
    (; zeta_spec, zeta, qfx, qfy, qflux_spec, B, B_spec) = locals

    flags = debug_flags()

    # momentum updated by fast terms
    uv_spec_new = vector_spec(
        (@. uv_spec_new.spheroidal = uv_spec.spheroidal + tau*duv_spec_fast.spheroidal),
        (@. uv_spec_new.toroidal = uv_spec.toroidal + tau*duv_spec_fast.toroidal),
    )
    (ux, uy) = uv = synthesis_vector!(uv, uv_spec_new, sph)

    # mass budgets
    fx = @. fx = -factor * ux * mass_air
    fy = @. fy = -factor * uy * mass_air
    flux_air_spec = analysis_vector!(flux_air_spec, erase(vector_spat(fx, fy)), sph)
    dmass_air_spec_slow = @. dmass_air_spec_slow = flux_air_spec.spheroidal * laplace

    fcx = @. fcx = -factor * ux * mass_consvar
    fcy = @. fcy = -factor * uy * mass_consvar
    flux_consvar_spec =
        analysis_vector!(flux_consvar_spec, erase(vector_spat(fcx, fcy)), sph)
    dmass_consvar_spec_slow = @. dmass_consvar_spec_slow = flux_consvar_spec.spheroidal * laplace

    # curl-form momentum budget
    zeta_spec = @. zeta_spec = -laplace * uv_spec_new.toroidal # curl
    zeta = synthesis_scalar!(zeta, zeta_spec, sph)

    qfx = @. qfx =  (flags.qU) * factor * (zeta + fcov) * uy
    qfy = @. qfy = -(flags.qU) * factor * (zeta + fcov) * ux
    qflux_spec = analysis_vector!(qflux_spec, erase(vector_spat(qfx, qfy)), sph)

    B = @. B = (factor / 2) * (ux^2 + uy^2)
    B_spec = analysis_scalar!(B_spec, erase(B), sph)
    duv_spec_slow = vector_spec(
        (@. duv_spec_slow.spheroidal = qflux_spec.spheroidal - (flags.gradB)*B_spec),
        (@. duv_spec_slow.toroidal = qflux_spec.toroidal),
    )

    locals = (; uv_spec_new, uv, fx, fy, flux_air_spec, fcx, fcy, flux_consvar_spec, 
                zeta_spec, zeta, qfx, qfy, qflux_spec, B, B_spec)

    return (; dmass_air_spec_slow, dmass_consvar_spec_slow, duv_spec_slow), locals
end


#========== hydrostatic balance and geopotential computation ==========
=======================================================================#

# Footgun: do not reassign to variable names if they are used inside the let ... end block, e.g.
#   p = similar!(p, mass)

function hydrostatic_pressure!(p_, model, mass::Array{Float64,3})
    p = similar!(p_, mass)
    @with model.mgr,
    let (irange, jrange) = (axes(p, 1), axes(p, 2))
        ptop, nz = model.vcoord.ptop, size(p, 3)
        half_metric = model.planet.radius^-2 / 2
        #= @inbounds =# for j in jrange
            @vec for i in irange
                p[i, j, nz] = ptop + half_metric * mass[i, j, nz]
                for k = nz:-1:2
                    p[i, j, k-1] =
                        p[i, j, k] + half_metric * (mass[i, j, k] + mass[i, j, k-1])
                end
            end
        end
    end
    return p
end

function Bernoulli!((B_, exner_, consvar_, Phi_), (mass_air, mass_consvar, p, uv), model)
    B = similar!(B_, p)
    exner = similar!(exner_, p)
    consvar = similar!(consvar_, p)
    Phi = @. Phi_ = model.Phis

    @with model.mgr,
    let (irange, jrange) = (axes(p, 1), axes(p, 2))
        flags = debug_flags()
        ux, uy = uv.ucolat, uv.ulon
        metric = model.planet.radius^-2
        Exner = model.gas(:p, :consvar).exner_functions
        #= @inbounds =# for j in jrange
            for k in axes(p, 3)
                @vec for i in irange
                    ke = (metric / 2) * (ux[i, j, k]^2 + uy[i, j, k]^2)
                    consvar_ijk = mass_consvar[i, j, k] / mass_air[i, j, k]
                    h, v, exner_ijk = Exner(p[i, j, k], consvar_ijk)
                    Phi_up = Phi[i, j] + metric * mass_air[i, j, k] * v # geopotential at upper interface
                    B[i, j, k] =
                        (flags.ke)*ke + (flags.Phi)*(Phi_up + Phi[i, j]) / 2 + (h - consvar_ijk * exner_ijk)
                    consvar[i, j, k] = consvar_ijk
                    exner[i, j, k] = exner_ijk
                    Phi[i, j] = Phi_up
                end
            end
        end
    end
    return B, exner, consvar, Phi
end

function Bernoulli_fast!((B_, exner_, consvar_, Phi_), (mass_air, mass_consvar, p), model)
    B = similar!(B_, p)
    exner = similar!(exner_, p)
    consvar = similar!(consvar_, p)
    Phi = @. Phi_ = model.Phis

    @with model.mgr,
    let (irange, jrange) = (axes(p, 1), axes(p, 2))
        flags = debug_flags()
        metric = model.planet.radius^-2
        Exner = model.gas(:p, :consvar).exner_functions
        #= @inbounds =# for j in jrange
            for k in axes(p, 3)
                @vec for i in irange
                    consvar_ijk = mass_consvar[i, j, k] / mass_air[i, j, k]
                    h, v, exner_ijk = Exner(p[i, j, k], consvar_ijk)
                    Phi_up = Phi[i, j] + metric * mass_air[i, j, k] * v # geopotential at upper interface
                    B[i, j, k] =
                        (flags.Phi)*(Phi_up + Phi[i, j]) / 2 + (h - consvar_ijk * exner_ijk)
                    consvar[i, j, k] = consvar_ijk
                    exner[i, j, k] = exner_ijk
                    Phi[i, j] = Phi_up
                end
            end
        end
    end
    return B, exner, consvar, Phi
end

#======================= low-level utilities ======================#

function compare(result, fun)
    res = deepcopy(result)
    alt = fun()
    @info "sameas" typeof(res) == typeof(alt)
    cmp = sameas(res, alt)
    @info "comparison" cmp
    error()
end
sameas(a::T, b::T) where {T<:Union{Tuple,NamedTuple}} = map(sameas, a, b)
sameas(a::T, b::T) where {T<:Array} = isapprox(a, b)

end # module Dynamics
