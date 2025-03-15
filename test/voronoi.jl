
Base.getindex(::ManagedLoops.DeviceManager, x::AbstractArray) = x

function setup_voronoi(sphere, choices, params, cpu)
    case = choices.TestCase(choices.precision; params.testcase...)
    conv(x::Number) = choices.precision(x)
    conv(x::Union{Tuple,NamedTuple}) = map(conv, x)
    params = merge(choices, conv(merge(case.params, params)))

    ## physical parameters needed to run the model
    @info case

    # stuff independent from initial condition
    gas = params.Fluid(params)
    vcoord = choices.coordinate(params.nz[1], params.ptop)

    surface_geopotential(lon, lat) = initial(case, lon, lat)[2]
    model = HPE(params, cpu, sphere, vcoord, surface_geopotential, gas)

    ## initial condition & standard diagnostics
    state = let
        init(lon, lat) = initial(case, lon, lat)
        init(lon, lat, p) = initial(case, lon, lat, p)
        CFHydrostatics.initial_HPE(init, model)
    end

    return model, diagnostics(model), state
end

function mintime(fun!, out_, mgr, args_)
    @info "$fun! on $mgr"
    args = args_ |> mgr
    out = out_ |> mgr
    function work()
        for _ in 1:10
            fun!(out, mgr, args...)
        end
        synchronize(mgr)
    end
    return minimum((@timed work()).time for _ in 1:10)
end

function bench(fun!, mgrs, args...)
    out = fun!(void, mgrs[1], args...)
    times = (mintime(fun!, out, mgr, args) for mgr in mgrs)
    return permutedims([fun!, times...])
end

mmul!(P, _, M, N) = mul!(similar!(P, M), M, N)
vexp!(y, mgr, x) = @. mgr[y] = @fastmath log(exp(x))
