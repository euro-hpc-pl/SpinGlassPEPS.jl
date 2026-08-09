# Task-scoped execution context: the device-memory budget that makes concurrent
# solves safe, and the truncation log that lets a heuristic contraction report
# how much weight it discarded.

using Base.ScopedValues: with

@testset "kernel_batch_size honours an explicit device budget" begin
    # Nothing installed by default, so nothing changes for existing callers.
    @test device_memory_budget() == 0

    # The budget is a *device* memory reservation, so it only governs the device
    # path. The CPU path has its own fixed budget and must ignore it — and so must
    # a caller that asks for `onGPU = true` on a machine with no working device,
    # since then there is no device memory to ration.
    @test with(() -> kernel_batch_size(Float64, 16, false), DEVICE_MEMORY_BUDGET => 1024) ==
          kernel_batch_size(Float64, 16, false)
    if !CUDA.functional()
        @test with(() -> kernel_batch_size(Float64, 16, true), DEVICE_MEMORY_BUDGET => 1024) ==
              kernel_batch_size(Float64, 16, true)
    end

    # Everything below is about the device path: skip it without a device rather
    # than assert device semantics that deliberately do not apply.
    if CUDA.functional()
        # With a budget in scope the batch is sized against a quarter of the
        # reservation — this is what stops N concurrent solves each batching as if
        # they owned the whole free pool.
        for T ∈ (Float32, Float64), per_item ∈ (1, 7, 4096)
            budget = 1 << 30   # 1 GiB reservation
            got = with(DEVICE_MEMORY_BUDGET => budget) do
                kernel_batch_size(T, per_item, true)
            end
            expected_max = (budget ÷ 4) ÷ (sizeof(T) * per_item)
            @test got <= expected_max
            @test got >= 1
            @test ispow2(got)
            # Halving the reservation must not increase the batch.
            smaller = with(DEVICE_MEMORY_BUDGET => budget ÷ 2) do
                kernel_batch_size(T, per_item, true)
            end
            @test smaller <= got
            # Float32 elements are half the size, so they get at least as large a
            # batch.
            if T === Float32
                f64 = with(DEVICE_MEMORY_BUDGET => budget) do
                    kernel_batch_size(Float64, per_item, true)
                end
                @test got >= f64
            end
        end

        # A tiny reservation must still yield a usable (>= 1) batch, not zero.
        @test with(() -> kernel_batch_size(Float64, 4096, true), DEVICE_MEMORY_BUDGET => 8) ==
              1

        # Budget of zero means "no explicit budget": fall back to the device query.
        fallback =
            with(() -> kernel_batch_size(Float64, 16, true), DEVICE_MEMORY_BUDGET => 0)
        @test fallback == kernel_batch_size(Float64, 16, true)
    end
end

@testset "svd_fact records discarded weight" begin
    # A rank-deficient matrix truncated below its rank: the discarded weight is
    # known in closed form from the singular values.
    A = Diagonal([4.0, 3.0, 2.0, 1.0]) |> Matrix
    log = TruncationLog()
    with(TRUNCATION_LOG => log) do
        svd_fact(A, 2, 0.0)
    end
    s = truncation_stats(log)
    total = 16.0 + 9.0 + 4.0 + 1.0
    @test s.count == 1
    @test s.discarded_sum ≈ (4.0 + 1.0) / total
    @test s.discarded_max ≈ s.discarded_sum
    @test s.saturated == 1          # the bond bound, not the tolerance, was binding
    @test s.dims_kept == 2
    @test s.dims_offered == 4

    # Retaining everything discards nothing, and is not counted as saturated.
    exact = TruncationLog()
    with(TRUNCATION_LOG => exact) do
        svd_fact(A, 4, 0.0)
    end
    e = truncation_stats(exact)
    @test e.count == 1
    @test e.discarded_sum < 1e-25
    @test e.saturated == 0
    @test e.dims_kept == e.dims_offered == 4

    # A tolerance that rejects the small singular values truncates without the
    # bond dimension being the binding constraint.
    tolerated = TruncationLog()
    with(TRUNCATION_LOG => tolerated) do
        svd_fact(A, 4, 0.6)   # keeps only σ = 4 (0.6 * 4 = 2.4 > 2)
    end
    t = truncation_stats(tolerated)
    @test t.dims_kept < 4
    @test t.saturated == 0
    @test t.discarded_sum > 0
end

@testset "truncation log accumulates and diffs" begin
    log = TruncationLog()
    A = Matrix(Diagonal([4.0, 3.0, 2.0, 1.0]))

    before = truncation_stats(log)
    @test before.count == 0

    with(TRUNCATION_LOG => log) do
        svd_fact(A, 2, 0.0)
        svd_fact(A, 3, 0.0)
    end
    after = truncation_stats(log)
    @test after.count == 2
    @test after.discarded_sum > 0

    # Snapshots subtract, so a caller can attribute error to one phase.
    mid = truncation_stats(log)
    with(TRUNCATION_LOG => log) do
        svd_fact(A, 1, 0.0)
    end
    phase = truncation_stats(log) - mid
    @test phase.count == 1
    @test phase.dims_kept == 1
    @test phase.discarded_sum ≈ truncation_stats(log).discarded_sum - mid.discarded_sum

    # Emptying resets every counter.
    @test truncation_stats(empty!(log)).count == 0
    @test truncation_stats(log).discarded_sum == 0

    # No log in scope: recording is a no-op and stats are still safe to ask for.
    @test record_truncation!(0.5, 1, 2, true) === nothing
    @test truncation_stats(nothing).count == 0
    @test TRUNCATION_LOG[] === nothing
end

@testset "truncation log is per task" begin
    # Concurrent solves must not pool their truncation error. Scoped values are
    # inherited by spawned tasks, so each solve installs its own.
    A = Matrix(Diagonal([4.0, 3.0, 2.0, 1.0]))
    a, b = TruncationLog(), TruncationLog()
    t1 = Threads.@spawn with(() -> svd_fact(A, 1, 0.0), TRUNCATION_LOG => a)
    t2 = Threads.@spawn with(() -> svd_fact(A, 3, 0.0), TRUNCATION_LOG => b)
    wait(t1)
    wait(t2)
    sa, sb = truncation_stats(a), truncation_stats(b)
    @test sa.count == 1
    @test sb.count == 1
    @test sa.dims_kept == 1
    @test sb.dims_kept == 3
    @test sa.discarded_sum > sb.discarded_sum
end

@testset "device peak tracker" begin
    p = DevicePeak()
    @test device_peak_bytes(p) == 0
    # Sampling with no tracker installed is a no-op.
    @test probe_device_peak!() === nothing
    @test DEVICE_PEAK_PROBE[] === nothing

    if CUDA.functional()
        tracker = DevicePeak()
        x = with(DEVICE_PEAK_PROBE => tracker) do
            y = CUDA.zeros(Float32, 64 * 1024 * 1024 ÷ 4)   # 64 MiB
            probe_device_peak!()
            y
        end
        @test device_peak_bytes(tracker) > 0
        x = nothing
        GC.gc(true)
        CUDA.reclaim()
    end
end
