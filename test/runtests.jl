const TEST_GROUPS = ("tensors", "networks", "exhaustive", "engine", "umbrella")

function selected_test_groups()
    selection = strip(get(ENV, "SPINGLASS_TEST_GROUP", "all"))
    isempty(selection) && (selection = "all")

    groups = selection == "all" ? collect(TEST_GROUPS) : strip.(split(selection, ','))
    unknown = setdiff(groups, TEST_GROUPS)
    isempty(unknown) || error(
        "Unknown SPINGLASS_TEST_GROUP value(s): $(join(unknown, ", ")). " *
        "Expected all or a comma-separated selection from $(join(TEST_GROUPS, ", ")).",
    )
    unique(groups)
end

groups = selected_test_groups()

# The umbrella checks cross-module reexports and an end-to-end solve, so keep it
# last whenever it is selected.
for group in TEST_GROUPS[1:end-1]
    group in groups || continue
    @info "Running SpinGlassPEPS test group" group
    include(joinpath(group, "runtests.jl"))
end

if "umbrella" in groups
    @info "Running SpinGlassPEPS test group" group = "umbrella"
    include("umbrella.jl")
end
