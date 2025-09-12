Base.getindex(::ManagedLoops.DeviceManager, x::AbstractArray) = x
mmul!(P, _, M, N) = mul!(similar!(P, M), M, N)
vexp!(y, mgr, x) = @. mgr[y] = @fastmath log(exp(x))

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
    GC.gc(true) # free GPU resources to avoid segfault ?
    return permutedims([fun!, times...])
end

function voronoi()
    choices = (gpu_blocks=(0, 32),
               precision=Float32,
               nz=(32, 96),
               # numerics
               meshname=DYNAMICO_meshfile("uni.2deg.mesh.nc"),
               coordinate=SigmaCoordinate,
               consvar=:temperature,
               TimeScheme=KinnmarkGray{2,5}, # RungeKutta4,
               # physics
               Fluid=IdealPerfectGas,
               TestCase=Jablonowski06)

    params = (
              # numerics
              courant=4.0,
              # physics
              radius=6.4e6,
              Omega=7.27220521664304e-5,
              ptop=225.52395239472398, # compatible with NCARL30 vertical coordinate
              Cp=1004.5,
              kappa=2 / 7,
              p0=1e5,
              T0=300,
              nu_gradrot=1e-16,
              hyperdiff_nu=0.002,
              # simulation
              testcase=(), # to override default test case parameters
              interval=6 * 3600)

    reader = DYNAMICO_reader(ncread, choices.meshname)
    vsphere = VoronoiSphere(reader; prec=choices.precision)
    @info vsphere

    mgrs = plain, simd = PlainCPU(), VectorizedCPU()
    mgr_names = ["CPU", "SIMD"]

    if oneAPI_functional
        @info "Functional oneAPI GPU detected !"
        oneAPI.versioninfo()
        gpu0 = LoopManagers.KernelAbstractions_GPU(oneAPIBackend(), (0, 0))
        gpu = LoopManagers.KernelAbstractions_GPU(oneAPIBackend(), choices.gpu_blocks)
        mgrs = (mgrs..., gpu0, gpu)
        mgr_names = ["CPU", "SIMD", "GPU", "GPU(blocked)"]
    end

    if CUDA_functional
        @info "Functional CUDA GPU detected !"
        CUDA.versioninfo()
        gpu0 = LoopManagers.KernelAbstractions_GPU(CUDABackend(), (0, 0))
        gpu = LoopManagers.KernelAbstractions_GPU(CUDABackend(), choices.gpu_blocks)
        mgrs = (mgrs..., gpu0, gpu)
        mgr_names = ["CPU", "SIMD", "GPU", "GPU(blocked)"]
    end

    model, diagnostics(model), state = setup_voronoi(vsphere,
                                                     (choices..., nz=choices.nz[1]), params,
                                                     simd)

    if Pkg.dependencies()[UUID("3699aaca-035b-4155-96ec-eecb526248de")].version >= v"0.3" # CFDomains
        for nz in choices.nz
            M = randn(choices.precision, 1024, 1024)
            N = randn(choices.precision, 1024, 1024)
            ue = randn(choices.precision, nz, length(edges(vsphere)))
            qi = randn(choices.precision, nz, length(cells(vsphere)))
            qv = randn(choices.precision, nz, length(duals(vsphere)))

            data = vcat(bench(vexp!, mgrs, M),
                        bench(mmul!, mgrs, M, N),
                        #                    bench(gradient!, mgrs, vsphere, qi),
                        #                    bench(gradperp!, mgrs, vsphere, qv),
                        #                    bench(perp!, mgrs, vsphere, ue),
                        #                    bench(curl!, mgrs, vsphere, ue),
                        bench(divergence!, mgrs, vsphere, ue),
                        bench(TRiSK!, mgrs, vsphere, ue))

            header = (["nz=$nz", mgr_names...])
            best = Highlighter((data, i, j) -> j > 1 &&
                                   all(data[i, k] >= data[i, j] for k in 2:size(data, 2)),
                               crayon"red bold")

            pretty_table(data;
                         header=header,
                         formatters=ft_printf("%7.6f", 2:5),
                         header_crayon=crayon"yellow bold",
                         highlighters=best,
                         tf=tf_unicode_rounded)
        end
    end
end
