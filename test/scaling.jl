function time(fun, N)
    fun()
    times = [(@timed fun()).time for i=1:(N+10)]
    sort!(times)
    return Float32(sum(times[1:N])/N)
end

function scaling(fun, name, simd, N::Int)
    @info "Multithread scaling for $name with $simd"
    @info "Threads \t elapsed \t speedup \t efficiency"
    single = 100f0
    for nt=1:Threads.nthreads()
        mgr = MultiThread(simd, nt)
        elapsed = time(N) do
            fun(mgr)
        end
        nt==1 && (single = elapsed)
        speedup = single/elapsed
        percent(x) = round(100*x; digits=0)
        @info "$nt \t\t $(round(elapsed; digits=4)) \t $(percent(speedup)) \t\t $(percent(speedup/nt))"
    end
end

function model(mgr=PlainCPU())
    ch, p = choices(), map(Float64, params())
    gas = ch.Fluid(merge(ch, p))
    (; mgr, gas, planet=(; radius=1.0), vcoord=(; ptop=1e3), Phis=0)
end

function scaling_pressure(choices)
    (; nlat, nz) = choices()
    mass = randn(2nlat, nlat, nz)
    p = hydrostatic_pressure!(void, model(), mass)
    for vsize in (16,32)
        scaling("hydrostatic_pressure!", VectorizedCPU(vsize), 100) do mgr
            hydrostatic_pressure!(p, model(mgr), mass)
        end
    end
end

function scaling_Bernouilli(choices)
    (; nlat, nz) = choices()
    mass = ones(2nlat, nlat, nz, 2)
    mass[:,:,:,2] = 300*mass[:,:,:,1] # theta=300K
    p = 1e5*ones(2nlat, nlat, nz)
    Phi = zeros(2nlat, nlat)
    B, exner, consvar = Bernoulli(void, void, void, Phis, model(), mass, p)
    for vsize in (16,32)
        scaling("Bernoulli!", VectorizedCPU(vsize), 100) do mgr
            Bernoulli!(B, exner, consvar, Phi, model(mgr), mass, p)
        end
    end
end
