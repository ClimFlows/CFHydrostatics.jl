import LinearAlgebra
using Random: MersenneTwister

function voronoi_adjoint_case()
    choices = (precision=Float64, coordinate=SigmaCoordinate, nz=5,
               consvar=:temperature, TestCase=Jablonowski06, Fluid=IdealPerfectGas)
    params = (ptop=100.0, radius=6.4e6, Omega=7.272e-5,
              Cp=1004.5, kappa=2/7, p0=1e5, T0=300.0,
              testcase=())
    reader = DYNAMICO_reader(ncread, VoronoiMeshes("uni.2deg.mesh.nc"))
    return setup_voronoi(VoronoiSphere(reader; prec=Float64), choices, params, PlainCPU())
end

field_dot(x, y) = sum(map(LinearAlgebra.dot, values(x), values(y)))

function check_voronoi_adjoint(reference, run!, output, state, args...)
    rng = MersenneTwister(137)
    direction = map(x -> randn(rng, size(x)) .* (LinearAlgebra.norm(x) / sqrt(length(x))), state)
    weights = map(x -> randn(rng, size(x)) ./ max(LinearAlgebra.norm(x), 1.0), output)
    shadow = make_zero(state)
    directional = ForwardDiff.derivative(0.0) do t
        input = map((x, dx) -> x .+ t .* dx, state, direction)
        field_dot(weights, reference(input))
    end
    autodiff(Enzyme.set_runtime_activity(Reverse), Const(run!), Const,
             Duplicated(deepcopy(output), deepcopy(weights)),
             Duplicated(deepcopy(state), shadow), args...)
    adjoint = field_dot(shadow, direction)
    @test isfinite(adjoint)
    @test adjoint ≈ directional rtol=2e-6 atol=1e-10
end

function voronoi_tendencies!(out, state, scratch, model)
    CFTimeSchemes.tendencies!(out, scratch, model, state, 0.0)
    return nothing
end

function voronoi_steps!(out, state, scratch, scheme, nsteps)
    dt = 30.0
    CFTimeSchemes.advance!(out, scheme, state, 0.0, dt, scratch)
    for step in 2:nsteps
        CFTimeSchemes.advance!(out, scheme, out, (step - 1) * dt, dt, scratch)
    end
    return nothing
end

function voronoi_steps(state, scheme, nsteps)
    out = map(similar, state)
    scratch = CFTimeSchemes.scratch_space(scheme, state, 0.0)
    voronoi_steps!(out, state, scratch, scheme, nsteps)
    return out
end

function voronoi_remap!(out, state, scratch, model)
    CFHydrostatics.RemapVoronoi.remap!(out, scratch, model, state)
    return nothing
end

function voronoi_steps_remap!(out, state, step_scratch, remap_scratch, scheme)
    voronoi_steps!(out, state, step_scratch, scheme, 3)
    voronoi_remap!(out, out, remap_scratch, scheme.model)
    return nothing
end

@testset "Voronoi HPE adjoint" begin
    model, _, state = voronoi_adjoint_case()
    tendencies, scratch = CFTimeSchemes.tendencies!(void, void, model, state, 0.0)
    @testset "Tendencies" begin
        check_voronoi_adjoint(voronoi_tendencies!, tendencies, state,
                              Duplicated(scratch, make_zero(scratch)), Const(model)) do input
            first(CFTimeSchemes.tendencies!(void, void, model, input, 0.0))
        end
    end

    scheme = RungeKutta4(model)
    @testset "RK4, $nsteps step(s)" for nsteps in (1, 3)
        scratch = CFTimeSchemes.scratch_space(scheme, state, 0.0)
        check_voronoi_adjoint(voronoi_steps!, state, state,
                              Duplicated(scratch, make_zero(scratch)), Const(scheme), Const(nsteps)) do input
            voronoi_steps(input, scheme, nsteps)
        end
    end

    @testset "Vertical remap" begin
        # Move layer interfaces away from the target pressure coordinate.
        factors = range(0.98, 1.02; length=size(state.masscov_air, 1))
        state.masscov_air .*= factors
        state.masscov_consvar .*= factors
        remap! = CFHydrostatics.RemapVoronoi.remap!
        remapped, scratch = remap!(void, void, model, state)
        check_voronoi_adjoint(voronoi_remap!, remapped, state,
                              Duplicated(scratch, make_zero(scratch)), Const(model)) do input
            first(remap!(void, void, model, input))
        end
        step_scratch = CFTimeSchemes.scratch_space(scheme, state, 0.0)
        check_voronoi_adjoint(voronoi_steps_remap!, remapped, state,
                              Duplicated(step_scratch, make_zero(step_scratch)),
                              Duplicated(scratch, make_zero(scratch)), Const(scheme)) do input
            future = voronoi_steps(input, scheme, 3)
            first(remap!(void, void, model, future))
        end
    end
end
