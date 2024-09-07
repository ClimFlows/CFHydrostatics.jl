module Dynamics

using MutatingOrNot: void, Void, similar

using ManagedLoops: @loops, @vec, no_simd
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

vector_spec(spheroidal, toroidal) = (; spheroidal, toroidal)
vector_spat(ucolat, ulon) = (; ucolat, ulon)
HPE_state(mass_air_spec, mass_consvar_spec, uv_spec) =
    (; mass_air_spec, mass_consvar_spec, uv_spec)

function tendencies!(dstate, scratch, model, state, t)
    # spectral fields are suffixed with _spec
    # vector, spectral = (spheroidal, toroidal)
    # vector, spatial = (ucolat, ulon)
    (; locals, locals_dmass, locals_duv) = scratch
    (; uv, mass_air, mass_consvar, p, B, exner, consvar, geopot) = locals
    (; mass_air_spec, mass_consvar_spec, uv_spec) = state
    dmass_air_spec, dmass_consvar_spec, duv_spec = dstate

    sph, invrad2, fcov = model.domain.layer, model.planet.radius^-2, model.fcov

    # flux-form mass balance
    (; dmass_air_spec, dmass_consvar_spec, uv, mass_air, mass_consvar), locals_dmass = let
        inputs = (; mass_air_spec, mass_consvar_spec, uv_spec)
        outputs = (; dmass_air_spec, dmass_consvar_spec, uv, mass_air, mass_consvar)
        fuse(mass_budgets!, outputs, scratch.locals_dmass, inputs, sph, sph.laplace, invrad2)
        # mass_budgets!(outputs, scratch.locals_dmass, inputs, sph, sph.laplace, invrad2)
    end

    p = hydrostatic_pressure!(p, model, mass_air)
    B, exner, consvar, geopot = Bernoulli!(
        B,
        exner,
        consvar,
        geopot,
        model,
        (air = mass_air, consvar = mass_consvar),
        p,
        uv,
    )

    # curl-form momentum balance
    (; duv_spec), locals_duv = let
        inputs = (; exner, consvar, B, uv_spec, uv)
        outputs = (; duv_spec)
        fuse(curl_form!, outputs, scratch.locals_duv, inputs, sph, sph.laplace, invrad2, fcov)
        # curl_form!(outputs, scratch.locals_duv, inputs, sph, sph.laplace, invrad2, fcov)
    end

    locals = (; uv, mass_air, mass_consvar, p, B, exner, consvar, geopot)
    scratch = (; locals, locals_duv, locals_dmass)

    return HPE_state(dmass_air_spec, dmass_consvar_spec, duv_spec), scratch
end

#=========================== flux-form mass budget ======================
       ∂Φ/∂t = -∇(Φu, Φv)
     uv is the momentum 1-form = a*(u,v)
     gh is the 2-form a²Φ
     divergence! is relative to the unit sphere
       => scale flux by radius^-2
========================================================================#

@inbounds function mass_budgets!(outputs, locals, inputs, sph, laplace, factor)
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
       => scale B and qflux by radius^-2
==================================================================#

function curl_form!(outputs, locals, inputs, sph, laplace, invrad2, fcov)
    (; exner, consvar, B, uv_spec, uv) = inputs
    (; duv_spec) = outputs
    (; qflux_spec, B_spec, fx, fy, zeta, zeta_spec, grad_exner, exner_spec) = locals

    exner_spec = analysis_scalar!(exner_spec, erase(exner), sph)
    (gradx, grady) = grad_exner = synthesis_spheroidal!(grad_exner, exner_spec, sph)
    zeta_spec = @. zeta_spec = -laplace * uv_spec.toroidal # curl
    zeta = synthesis_scalar!(zeta, zeta_spec, sph)
    (ux, uy) = uv
    fx = @. fx =  invrad2 * (zeta + fcov) * uy - consvar * gradx
    fy = @. fy = -invrad2 * (zeta + fcov) * ux - consvar * grady
    qflux_spec = analysis_vector!(qflux_spec, erase(vector_spat(fx, fy)), sph)

    B_spec = analysis_scalar!(B_spec, erase(B), sph)
    duv_spec = vector_spec(
        (@. duv_spec.spheroidal = qflux_spec.spheroidal - B_spec),
        (@. duv_spec.toroidal = qflux_spec.toroidal),
    )
    return (; duv_spec),
    (; qflux_spec, B_spec, fx, fy, zeta, zeta_spec, grad_exner, exner_spec)
end

#========== hydrostatic balance and geopotential computation ==========
=======================================================================#

function hydrostatic_pressure!(p, model, air::Array{Float64,3})
    p = similar(air, p)
    compute_hydrostatic_pressure(model.mgr, p, model, air)
    return p
end

@loops function compute_hydrostatic_pressure(_, p, model, mass)
    let (irange, jrange) = (axes(p, 1), axes(p, 2))
        ptop, nz = model.vcoord.ptop, size(p, 3)
        half_invrad2 = model.planet.radius^-2 / 2
        for j in jrange
            @vec for i in irange
                p[i, j, nz] = ptop + half_invrad2 * mass[i, j, nz]
                for k = nz:-1:2
                    p[i, j, k-1] =
                        p[i, j, k] + half_invrad2 * (mass[i, j, k] + mass[i, j, k-1])
                end
            end
        end
    end
end

function Bernoulli!(B, exner, consvar, Phi, model, masses, p, uv)
    # similar(x,y) allocates only if y::Void
    B = similar(p, B)
    exner = similar(p, exner)
    consvar = similar(p, consvar)

    Phi = @. Phi = model.Phis
    compute_Bernoulli!(model.mgr, B, exner, consvar, Phi, masses, p, uv, model)
    return B, exner, consvar, Phi
end

@loops function compute_Bernoulli!(_, B, exner, consvar, Phi, masses, p, uv, model)
    let (irange, jrange) = (axes(p, 1), axes(p, 2))
        ux, uy = uv.ucolat, uv.ulon
        invrad2 = model.planet.radius^-2
        Exner = model.gas(:p, :consvar).exner_functions
        @inbounds for j in jrange
            for k in axes(p, 3)
                @vec for i in irange
                    ke = (invrad2 / 2) * (ux[i, j, k]^2 + uy[i, j, k]^2)
                    consvar_ijk = masses.consvar[i, j, k] / masses.air[i, j, k]
                    h, v, exner_ijk = Exner(p[i, j, k], consvar_ijk)
                    Phi_up = Phi[i, j] + invrad2 * masses.air[i, j, k] * v # geopotential at upper interface
                    B[i, j, k] =
                        ke + (Phi_up + Phi[i, j]) / 2 + (h - consvar_ijk * exner_ijk)
                    consvar[i, j, k] = consvar_ijk
                    exner[i, j, k] = exner_ijk
                    Phi[i, j] = Phi_up
                end
            end
        end
    end
end

#======================= low-level utilities ======================#

@inline function fuse(fun::Fun, outputs, locals, inputs, sph, other...) where {Fun}
    if hasvoid(outputs) || hasvoid(locals)
        fuse_hasvoid(fun, outputs, locals, inputs, sph, other...)
    else
        fuse_novoid(fun, outputs, locals, inputs, sph, other...)
    end
end

@inline function fuse_hasvoid(fun::Fun, outputs, locals, inputs, sph, other...) where {Fun}
    outputs, locals = fun(outputs, locals, inputs, sph, other...)
    local_slices = [Slicer(k)(locals) for k = 1:length(sph.ptrs)]
    return outputs, local_slices
end

@inline function fuse_novoid(fun::Fun, outputs, locals, inputs, sph, other...) where {Fun}
    n = size(inputs[1])[end]                # last dimension of first input array
    batch(sph, n, 1) do sph_, thread, k, _   # let SHTnsSpheres manage threads
        locs = locals[thread]
        ins = Viewer(k)(inputs)
        outs = Viewer(k)(outputs)
        fun(outs, locs, ins, sph_, other...)
    end
    return outputs, locals
end

hasvoid(_) = false
hasvoid(::Void) = true
hasvoid(tup::NamedTuple) = any(hasvoid, tup)

abstract type Recurser end
(rec::Recurser)(x::Union{Tuple,NamedTuple}) = map(rec, x)
(rec::Recurser)(x, y, z...) = s((x, y, z...))

struct Viewer <: Recurser
    k::Int
end
(s::Viewer)(a::Array{Float64,3}) = view(a, :, :, s.k)
(s::Viewer)(a::Matrix{ComplexF64}) = view(a, :, s.k)

struct Slicer <: Recurser
    k::Int
end
(s::Slicer)(x::Array{Float64,3}) = x[:, :, s.k]
(s::Slicer)(x::Matrix{ComplexF64}) = x[:, s.k]

abstract type Copier end
@inline (cop::Copier)(x::NamedTuple{names}, y::NamedTuple{names}) where {names} =
    cop(Tuple(x), Tuple(y))
@inline (cop::Copier)(::Tuple{}, ::Tuple{}) = nothing
@inline function (cop::Copier)(x::Tuple, y::Tuple)
    xx, xxx... = x
    yy, yyy... = y
    cop(xx, yy)
    cop(xxx, yyy)
    nothing
end

struct CopyIn! <: Copier
    k::Int
end
@inline (cop::CopyIn!)(lhs::Matrix{Float64}, rhs::Array{Float64,3}) =
    copyin!(lhs, rhs, cop.k)
@inline (cop::CopyIn!)(lhs::Vector{ComplexF64}, rhs::Matrix{ComplexF64}) =
    copyin!(lhs, rhs, cop.k)
@inline copyin!(lhs, rhs, k) = @inbounds for i in eachindex(lhs)
    lhs[i] = rhs[i+(k-1)*length(lhs)]
end

struct CopyOut! <: Copier
    k::Int
end
@inline (cop::CopyOut!)(lhs::Array{Float64,3}, rhs::Matrix{Float64}) =
    copyout!(lhs, rhs, cop.k)
@inline (cop::CopyOut!)(lhs::Matrix{ComplexF64}, rhs::Vector{ComplexF64}) =
    copyout!(lhs, rhs, cop.k)
@inline copyout!(lhs, rhs, k) = @inbounds for i in eachindex(rhs)
    lhs[i+(k-1)*length(rhs)] = rhs[i]
end

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
