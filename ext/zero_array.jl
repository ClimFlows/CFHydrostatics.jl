struct Zero <: Real end

struct ZeroArray{N} <: AbstractArray{Zero,N}
    ax::NTuple{N, Base.OneTo{Int}}
end
@inline Base.axes(z::ZeroArray) = z.ax
@inline Base.getindex(::ZeroArray, i...) = Zero()

@inline Base.:*(::Number, ::Zero) = Zero()
@inline Base.:*(::Zero, ::Number) = Zero()
@inline Base.:*(::Zero, ::Zero) = Zero()

@inline Base.:+(x::Complex, ::Zero) = x
@inline Base.:+(x::Number, ::Zero) = x
@inline Base.:+(::Zero, x::Number) = x
@inline Base.:+(::Zero, x::Complex) = x
@inline Base.:+(::Zero, ::Zero) = Zero()
