using ArgParse
using PythonOT
solvers = Dict(
    "lamp" => LAMP,
    "pdmp" => PDMP,
    "sinkhorn" => sinkhorn_log,
    "annealed_sinkhorn" => annealed_sinkhorn_log,
    "greenkhorn" => greenkhorn_log,
    "apdamd" => APDAMD,
    "apdagd" => APDAGD,
    "hpd" => HPD,
    "acc_sinkhorn" => accelerated_sinkhorn,
    "dextrap" => dual_extrapolation
)
ctransfer_solvers = Dict(
    "lamp" => LAMP,
    "sinkhorn" => sinkhorn_log
)
cellsim_solvers = Dict(
    "lamp" => extragradient_cell_distance,
    "sinkhorn" => sinkhorn_cell_distance,
    "annealed_sinkhorn" => annealed_sinkhorn_cell_distance,
)
settings = ArgParseSettings(prog="culamp")
@add_arg_table! settings begin
    "run"
    help = "Run discrete OT problem"
    action = :command

    "ctransfer"
    help = "Perform color transfer using a metric kernel. CUDA is used by default."
    action = :command

    "cellsim"
    help = "Compute a streaming C-index over OT-scOmics cells using an on-the-fly feature cost kernel."
    action = :command

end
@add_arg_table! settings["run"] begin
    "--algorithm", "-a"
    help = "Algorithm to solve the DOT instance. Options are: $(join(keys(solvers), ", "))"
    range_tester = x -> x in keys(solvers)
    default = "sinkhorn"
    "--settings"
    help = "Solver configuration settings"
    default = "./test.json"
    "--cuda"
    help = "Use CUDA"
    action = :store_true
    "--p"
    help = "p for distance computation (>= 10 for infinity norm, 0 for uniform cost)"
    arg_type = Float64
    default = 2.0
    "--weights"
    help = "Path to CSV-formatted weight matrix"
    default = ""
    "--kernel"
    help = "Use kernels to compute OT matrices on the fly (supported: lamp, sinkhorn, annealed_sinkhorn)"
    action = :store_true
    "--frequency"
    help = "Printing frequency"
    default = 100
    arg_type = Int
    "file1"
    help = "Path to target DOTMark-formatted file (row marginal) (TODO: Add support for more input types)"
    required = true
    "file2"
    help = "Path to target DOTMark-formatted file (col marginal) (TODO: Add support for more input types)"
    required = true
    "--output1"
    help = "Output path for assignment 1"
    default = ""
    "--output2"
    help = "Output path for assignment 2"
    default = ""
    "--height"
    help = "Image height"
    default = typemax(Int)
    arg_type = Int
    "--width"
    help = "Image width"
    default = typemax(Int)
    arg_type = Int
    "--fp32"
    help = "Enable Float32 computation (otherwise uses Float64)"
    action = :store_true
    "--potential-out"
    help = "Output path for dual potentials. Order is (1) Simplex dual (if using extragradient), (2) Potential for Row Marginal, (3) Potential for Column Marginal>"
    default = ""
end
@add_arg_table! settings["ctransfer"] begin
    "--algorithm", "-a"
    help = "Algorithm to solve the color transfer instance. Options are: $(join(keys(ctransfer_solvers), ", "))"
    range_tester = x -> x in keys(ctransfer_solvers)
    default = "sinkhorn"
    "--settings"
    help = "Solver configuration settings"
    default = "./test.json"
    "--frequency"
    help = "Printing frequency"
    default = 100
    arg_type = Int
    "--p"
    help = "p for distance computation (>= 10 for infinity norm, 0 for uniform cost)"
    arg_type = Float64
    default = 2.0
    range_tester = x -> x >= 0
    "--height"
    help = "Image height"
    default = 128
    arg_type = Int
    "--width"
    help = "Image width"
    default = 128
    arg_type = Int
    "file1"
    help = "Path to target input image file (row marginal)"
    required = true
    "file2"
    help = "Path to target input image file (column marginal)"
    required = true
    "--output1"
    help = "Output path for color mapped image 1"
    required = true
    "--output2"
    help = "Output path for color mapped image 2"
    required = true
end
@add_arg_table! settings["cellsim"] begin
    "--algorithm", "-a"
    help = "Algorithm to solve the cell OT problem. Options are: $(join(keys(cellsim_solvers), ", "))"
    range_tester = x -> x in keys(cellsim_solvers)
    default = "lamp"
    "--settings"
    help = "Solver configuration settings"
    default = "./test.json"
    "--frequency"
    help = "Progress printing frequency in number of cell pairs"
    default = 100
    arg_type = Int
    "--cost"
    help = "Ground cost metric over features. Options are: $(join(CELL_COST_METRICS, ", "))"
    range_tester = x -> lowercase(x) in CELL_COST_METRICS
    default = "correlation"
    "--normalize-features"
    help = "L1-normalize each feature across cells before kernel distances are evaluated"
    default = true
    arg_type = Bool
    "--num-features"
    help = "Number of features to include"
    default = typemax(Int)
    arg_type = Int
    "--num-subproblems"
    help = "Number of subproblems to run (limits pairwise comparisons, just for benchmarking)"
    default = typemax(Int)
    arg_type = Int
    "--seed"
    help = "Random seed for cell selection"
    default = 0
    arg_type = Int
    "--num-cells"
    help = "Number of cells to include"
    default = typemax(Int)
    arg_type = Int
    "--output"
    help = "Optional output path for a one-line summary"
    default = ""
    "file"
    help = "Path to an OT-scOmics preprocessed CSV or CSV.GZ file"
    required = true
end
function run_dot(parsed_args)
    args = read_args_json(parsed_args["settings"])
    size = (parsed_args["height"], parsed_args["width"])
    if args["fp32"]
        dtype = Float32
    else
        dtype = Float64
    end
    marginal1, h, w, N = read_dotmark_data(parsed_args["file1"], size; dtype=dtype)
    marginal2, h2, w2, N2 = read_dotmark_data(parsed_args["file2"], size; dtype=dtype)
    @assert h == h2 && w == w2 && N == N2
    # mix it with a little bit of the uniform distribution for stability
    r = normalize(marginal1 .+ 1e-6, 1)
    c = normalize(marginal2 .+ 1e-6, 1)
    if !parsed_args["kernel"]
        if parsed_args["weights"] != ""
            W = read_weights(parsed_args["weights"], dtype=dtype)
        else
            W = get_euclidean_distance(h, w; p=parsed_args["p"], dtype=dtype)
        end
        W∞ = norm(W, Inf)
        # args.eta_p /= W∞
        # args.epsilon /= W∞
        W ./= W∞
        # W = (W .+ 1) / 2
        # W .= rand(rng, N, N2)
        if parsed_args["cuda"]
            r, c, W = map(CuArray, [r, c, W])
        end
        solvers[parsed_args["algorithm"]](r, c, W, args, parsed_args["frequency"])
    else
        r = CuArray(r)
        c = CuArray(c)
        locations = zeros(Float64, 3, h * w)
        for i in 1:(h*w)
            locations[1, i] = (i - 1) ÷ w
            locations[2, i] = (i - 1) % w
        end
        locations = CuArray(locations)
        # locations2 = CuArray(locations)
        if parsed_args["algorithm"] == "sinkhorn"
            sinkhorn_euclidean(r, c, locations, locations, parsed_args["output1"], parsed_args["output2"], parsed_args["potential-out"], args, parsed_args["frequency"], parsed_args["p"])
        elseif parsed_args["algorithm"] == "annealed_sinkhorn"
            annealed_sinkhorn_euclidean(r, c, locations, locations, parsed_args["output1"], parsed_args["output2"], parsed_args["potential-out"], args, parsed_args["frequency"], parsed_args["p"])
        elseif parsed_args["algorithm"] == "lamp"
            extragradient_euclidean(r, c, locations, locations, parsed_args["output1"], parsed_args["output2"], parsed_args["potential-out"], args, parsed_args["frequency"], parsed_args["p"])
        end
    end
end

function run_ctransfer(parsed_args)
    args = read_args_json(parsed_args["settings"])
    size = (parsed_args["height"], parsed_args["width"])
    if parsed_args["algorithm"] == "lamp"
        extragradient_color_transfer(parsed_args["file1"], parsed_args["file2"], parsed_args["output1"], parsed_args["output2"], size, args, parsed_args["frequency"], parsed_args["p"])
    elseif parsed_args["algorithm"] == "sinkhorn"
        sinkhorn_color_transfer(parsed_args["file1"], parsed_args["file2"], parsed_args["output1"], parsed_args["output2"], size, args, parsed_args["frequency"], parsed_args["p"])
    end
end

function run_cellsim(parsed_args)
    args = read_args_json(parsed_args["settings"])
    metric = normalize_cell_cost_metric(parsed_args["cost"])
    solver = cellsim_solvers[parsed_args["algorithm"]]
    result = compute_otscomics_c_index(
        parsed_args["file"],
        parsed_args["num-features"],
        parsed_args["num-subproblems"],
        parsed_args["num-cells"],
        args;
        metric=metric,
        normalize_features=parsed_args["normalize-features"],
        frequency=parsed_args["frequency"],
        solver=solver,
        seed=parsed_args["seed"]
    )
    summary = "c_index=$(result.c_index),Sw=$(result.Sw),Smin=$(result.Smin),Smax=$(result.Smax),Nw=$(result.Nw),pairs=$(result.pairs)"
    println(summary)
    if parsed_args["output"] != ""
        open(parsed_args["output"], "w") do io
            println(io, summary)
        end
    end
    return result
end


function run_from_arguments(arguments::Vector{String})
    parsed_args = parse_args(arguments, settings)
    if parsed_args["%COMMAND%"] == "run"
        run_dot(parsed_args["run"])
    elseif parsed_args["%COMMAND%"] == "ctransfer"
        run_ctransfer(parsed_args["ctransfer"])
    elseif parsed_args["%COMMAND%"] == "cellsim"
        run_cellsim(parsed_args["cellsim"])
    end
end


if abspath(PROGRAM_FILE) == @__FILE__
    run_from_arguments(ARGS)
end
