struct Fused{Fun}
    fun::Fun
end

@inline function (fused::Fused)(outputs, locals, inputs, sph, other...)
    if hasvoid(outputs) || hasvoid(locals)
        fuse_hasvoid(fused.fun, outputs, locals, inputs, sph, other...)
    else
        fuse_novoid(fused.fun, outputs, locals, inputs, sph, other...)
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

