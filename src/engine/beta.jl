# beta.jl: inverse-temperature ladders with warm-started boundary MPS.
#
# β is the solver's most consequential and least guidable parameter. It sets how
# sharply the Boltzmann distribution the PEPS network represents concentrates on
# low-energy states: too small and the conditional probabilities the search
# branches on carry little information about the ground state. The published
# interface takes a single scalar and its own documentation concedes that "the
# optimal value of β often depends on the problem instance" — leaving the user to
# guess, or to hand-scan, with no feedback either way.
#
# What the *upper* end costs is less obvious than it looks, and is worth stating
# because the opposite is easy to assume. Discarded weight is **not** monotone in
# β. Measured on ten 2500-spin square-lattice instances (bond 8, 500 states):
#
#     β        2        3        4        6        8
#     Σε    2.3e-4   2.2e-3   3.4e-3   7.3e-4   1.4e-4
#     err   7.8e-4   2.1e-4   6.2e-5      0        0
#
# Σε rises while the distribution is sharpening — there is more structure for the
# boundary MPS to carry — and falls again once it concentrates enough to be close
# to a product state, which truncates easily. Solution quality, meanwhile, keeps
# improving. So on that family the *best* βs have among the *lowest* discarded
# weight, and a guard on Σε alone would have rejected the good rungs and kept the
# bad ones. Σε bounds how much of the distribution the contraction threw away; it
# is not a proxy for whether the search then found a good state.
#
# Two pieces address the β problem here:
#
#   * `beta_ladder` walks an increasing schedule of βs. Each step reuses the
#     previous step's boundary MPS as the starting point for variational
#     compression rather than rebuilding `W * ψ` exactly and truncating it, so a
#     ladder costs materially less than the same number of independent solves.
#   * every step reports the energy it reached and the truncation error it
#     accumulated (see `TruncationLog`), so a scan produces evidence rather than a
#     single number. Select on energy; read Σε as a separate statement about how
#     well the contraction represented the distribution at that β.

export set_beta!, beta_ladder, BetaStepReport, BetaLadderSolution, selected_solution

using Base.ScopedValues: with

"""
$(TYPEDSIGNATURES)

Re-target `ctr` at a new inverse temperature.

Every cached quantity — MPO layers, boundary MPS, environments — depends on β, so
all of it is evicted. When `warm_start` is `true` the boundary MPS retained by a
previous `low_energy_spectrum(...; retain_mps = true)` are kept and will be used
as variational starting points for the corresponding rows at the new β; when it
is `false` they are dropped and the next solve builds every row from scratch.
"""
function set_beta!(
    ctr::MpsContractor{T,R,S},
    beta::Real;
    warm_start::Bool = true,
) where {T,R,S}
    warm_start || empty!(ctr.guess)
    empty!(ctr.cache)
    empty!(ctr.peps.lp, :GPU)
    ctr.beta = S(beta)
    ctr
end

"""
$(TYPEDSIGNATURES)

One rung of a [`beta_ladder`](@ref).

# Fields
- `beta::Float64`: the inverse temperature solved at.
- `energy::Float64`: lowest energy found (`NaN` if the step failed).
- `wall_time::Float64`: seconds spent on this step.
- `truncation::TruncationStats`: weight discarded by this step's contraction.
- `warm_started::Bool`: whether this step started from the previous step's
  boundary MPS.
- `trusted::Bool`: whether the accumulated discarded weight stayed within the
  guard (`max_discarded`). An untrusted step's energy is reported but is not
  eligible to be selected.
- `error`: the exception if the step failed, otherwise `nothing`.
"""
struct BetaStepReport
    beta::Float64
    energy::Float64
    wall_time::Float64
    truncation::TruncationStats
    warm_started::Bool
    trusted::Bool
    error::Any
end

"""
$(TYPEDSIGNATURES)

Result of a [`beta_ladder`](@ref).

# Fields
- `betas::Vector{Float64}`: the schedule, in the order solved.
- `solutions::Vector{Union{Nothing,Solution}}`: aligned with `betas`.
- `selected_index::Int`: the rung chosen — lowest energy among the rungs whose
  contraction was trusted, or `0` if none produced a solution.
- `steps::Vector{BetaStepReport}`
"""
struct BetaLadderSolution
    betas::Vector{Float64}
    solutions::Vector{Union{Nothing,Solution}}
    selected_index::Int
    steps::Vector{BetaStepReport}
end

"""
$(TYPEDSIGNATURES)

The [`Solution`](@ref) chosen by a β ladder. Throws if every rung failed.
"""
function selected_solution(l::BetaLadderSolution)
    l.selected_index == 0 && error("every rung of the beta ladder failed")
    l.solutions[l.selected_index]
end

function Base.show(io::IO, l::BetaLadderSolution)
    sel = l.selected_index
    print(
        io,
        "BetaLadderSolution($(length(l.betas)) rungs, ",
        "selected=",
        sel == 0 ? "none" : "β=$(l.betas[sel]) E=$(round(l.steps[sel].energy, sigdigits=8))",
        ", trusted=$(count(s -> s.trusted, l.steps))/$(length(l.steps)))",
    )
end

"""
$(TYPEDSIGNATURES)

Solve one instance across an increasing schedule of inverse temperatures,
reusing each step's boundary MPS to warm-start the next, and select the
lowest-energy result among the steps that stayed within the error guard.

Selection is on **energy**. The guard only excludes rungs whose contraction
discarded more than `max_discarded`; it is not a quality ranking, and by default
(`Inf`) it excludes nothing.

The contractor is mutated in place (its β is changed and its caches evicted at
every step), so pass a contractor this call may own.

# Arguments
- `ctr::MpsContractor`: the contractor to re-target at each β.
- `betas`: the schedule. Should be increasing — warm-starting only helps when
  consecutive βs are close. A non-increasing schedule is accepted with a warning.
- `sparams::SearchParameters`

# Keyword arguments
- `merge_strategy = _ -> no_merge`: `ctr -> strategy`, as in
  [`sweep_transformations`](@ref).
- `symmetry::Symbol = :noZ2`
- `warm_start::Bool = true`: reuse the previous rung's boundary MPS.
- `max_discarded = Inf`: guard on accumulated discarded weight
  (`TruncationStats.discarded_sum`). A rung exceeding it is marked untrusted and
  excluded from selection; with the default no rung is ever excluded.

  !!! warning
      This guard is only meaningful with `warm_start = false`. A cold build forms
      `W * ψ` exactly and truncates it, so the discarded weight is recorded; a
      warm start optimizes within a fixed bond dimension and never performs a
      truncating factorization, so it reports ~zero discarded weight whatever its
      accuracy — its error is an optimization gap the truncation log does not
      measure. Setting both warns.
- `stop_when_untrusted::Bool = false`: stop climbing once a rung is untrusted.
  Use with care: discarded weight is **not** monotone in β (see the file header),
  so a later rung may well come back under the guard — and on the family measured
  there, the rungs with the lowest discarded weight were the ones that found the
  best energies. This option is for bounding cost when a rung blows up, not for
  locating the best β.
- `show_progress::Bool = false`

# Returns
A [`BetaLadderSolution`](@ref); [`selected_solution`](@ref) gives the winner.

# Example
```julia
ladder = beta_ladder(ctr, [0.5, 1.0, 2.0, 4.0], search_params;
                     max_discarded = 1e-3, stop_when_untrusted = true)
sol = selected_solution(ladder)
[(s.beta, s.energy, s.truncation.discarded_sum) for s ∈ ladder.steps]
```
"""
function beta_ladder(
    ctr::MpsContractor{T,R,S},
    betas,
    sparams::SearchParameters;
    merge_strategy = _ -> no_merge,
    symmetry::Symbol = :noZ2,
    warm_start::Bool = true,
    max_discarded::Real = Inf,
    stop_when_untrusted::Bool = false,
    show_progress::Bool = false,
) where {T,R,S}
    schedule = Float64[Float64(b) for b ∈ betas]
    isempty(schedule) && throw(ArgumentError("beta schedule is empty"))
    any(b -> b <= 0, schedule) &&
        throw(ArgumentError("inverse temperatures must be positive, got $schedule"))
    issorted(schedule) || @warn """
    Beta schedule is not increasing: $schedule.
    Warm starting assumes consecutive betas are close, and the error guard is
    read as how far up the ladder the contraction survived.
    """

    # The discarded-weight guard cannot see a warm-started row's error. A cold
    # build forms W * ψ exactly and then truncates, so the weight it drops is
    # recorded by `svd_fact`; a warm start instead optimizes within a fixed bond
    # dimension and never performs a truncating factorization, so it reports
    # ~zero discarded weight whatever its actual accuracy — the error is an
    # optimization gap, which the truncation log does not measure. Combining the
    # two would rate every warm-started rung as perfectly trustworthy.
    if warm_start && isfinite(max_discarded)
        @warn """
        `max_discarded` is set together with `warm_start = true`: the guard will
        not be meaningful for warm-started rungs. A warm start optimizes within a
        fixed bond dimension and never performs a truncating factorization, so it
        reports ~zero discarded weight regardless of its accuracy. Re-run with
        `warm_start = false` to audit contraction error across the schedule.
        """
    end

    solutions = Vector{Union{Nothing,Solution}}(nothing, length(schedule))
    steps = BetaStepReport[]

    for (k, beta) ∈ enumerate(schedule)
        # First rung has nothing to warm start from; later rungs reuse what the
        # previous `retain_mps = true` solve left behind.
        warmed = warm_start && k > 1 && !isempty(ctr.guess)
        set_beta!(ctr, beta; warm_start = warmed)
        log = TruncationLog()
        t0 = time()
        try
            sol, _ = with(TRUNCATION_LOG => log) do
                low_energy_spectrum(
                    ctr,
                    sparams,
                    merge_strategy(ctr),
                    symmetry;
                    show_progress = show_progress,
                    # No point snapshotting on the last rung: nothing will
                    # consume it, and the snapshot is one host-side MPS per row.
                    retain_mps = warm_start && k < length(schedule),
                )
            end
            stats = truncation_stats(log)
            trusted = stats.discarded_sum <= max_discarded
            solutions[k] = sol
            push!(
                steps,
                BetaStepReport(
                    beta,
                    isempty(sol.energies) ? NaN : Float64(first(sol.energies)),
                    time() - t0,
                    stats,
                    warmed,
                    trusted,
                    nothing,
                ),
            )
            if !trusted && stop_when_untrusted
                @info """
                Stopping the beta ladder at β=$beta: discarded weight \
                $(stats.discarded_sum) exceeds the guard $max_discarded. \
                Note that discarded weight is not monotone in β — a higher rung may \
                fall back under the guard — so this stops on cost, not on a \
                conclusion that higher β cannot help.
                """
                break
            end
        catch err
            err isa InterruptException && rethrow()
            @error "beta ladder rung β=$beta failed" exception =
                (err, catch_backtrace())
            push!(
                steps,
                BetaStepReport(
                    beta,
                    NaN,
                    time() - t0,
                    truncation_stats(log),
                    warmed,
                    false,
                    err,
                ),
            )
            # A failed rung leaves the contractor's warm-start slot in an unknown
            # state; drop it so the next rung builds cleanly.
            empty!(ctr.guess)
        end
    end

    # Release the warm-start snapshots; the ladder is over and they are one
    # host-resident MPS per row.
    empty!(ctr.guess)

    # Select the lowest energy among trusted rungs. If the guard rejected
    # everything, fall back to all rungs rather than returning nothing: the
    # caller can see from `steps` that nothing was trusted and decide for itself.
    selected = _select_rung(steps, true)
    selected == 0 && (selected = _select_rung(steps, false))
    BetaLadderSolution(schedule, solutions, selected, steps)
end

# Rungs are appended in schedule order and the ladder only ever stops early, so a
# step's position in `steps` is its position in the schedule.
function _select_rung(steps::Vector{BetaStepReport}, require_trusted::Bool)
    best, best_i = Inf, 0
    for (i, s) ∈ enumerate(steps)
        isnan(s.energy) && continue
        require_trusted && !s.trusted && continue
        if s.energy < best
            best, best_i = s.energy, i
        end
    end
    best_i
end
