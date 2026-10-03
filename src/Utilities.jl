import CSV
using DataFrames
using JSON3
using IterTools
using StructTypes
using ArgParse
# using DelimitedFiles
using Images

@kwdef struct EOTProblem{TA,TM,R}
    η::R
    r::TA
    c::TA
    W::TM
    b::TA = vcat(r, c)
    N = size(r, 1)
end
@kwdef mutable struct EOTArgs{R}
    eta_p::R = 0.0
    eta_mu::R = 0.0
    tau_p::R = 1.0
    tau_mu::R = 1.0
    alpha::R = 0.01
    B::R = 1.0
    epsilon::Real = 1e-4
    anneal_mult::R = 0.95
    itermax::Int = 10_000
    inner_iter::Int = 10
    tmax::Float64 = Inf
    verbose::Bool = true
end
StructTypes.StructType(::Type{EOTArgs}) = StructTypes.Mutable()

function logsumexp(x::AbstractArray{T}, dims=Nothing) where T<:Number
    if dims == Nothing
        dims = 1:length(size(x))
    end
    maxx = maximum(x, dims=dims)
    v1 = log.(sum(exp.(x .- maxx), dims=dims)) .+ maxx
    return v1
end
function logsumexp!(out::AbstractArray{T}, maxcache::AbstractArray{T}, x::AbstractArray{T}, dims=Nothing) where T<:Number
    maximum!(maxcache, x)
    sum!(out, exp.(x .- maxcache))
    out .= log.(out) .+ maxcache
end
function Zvals(x::AbstractArray{T}; dims=[]) where T
    maxx = maximum(x, dims=dims)
    return sum(exp.(x .- maxx), dims=dims)
end

@inline
function dual_gradient!(output::TA, x::TA, prob::EOTProblem) where TA
    grad_cache1 = softmax(-prob.W / prob.η .+ x[1:prob.N] .+ x[(prob.N+1):end]', dims=2)
    grad_cache2 = softmax(-prob.W / prob.η .+ x[1:prob.N] .+ x[(prob.N+1):end]', dims=1)'
    output .= vcat(grad_cache1, grad_cache2)
    output .-= prob.b
end

function KL(theta1, theta2, eta1, eta2, r, W, W∞)
    p1 = softmax(-(W * 0.5 / W∞ .+ theta1') ./ eta1, norm_dims=2)
    p2 = softmax(-(W * 0.5 / W∞ .+ theta2') ./ eta2, norm_dims=2)
    return dot(r .* p1, log.(p1 .+ 1e-30) - log.(p2 .+ 1e-30))
end
function KL(p1, p2, r)
    return dot(r .* p1, log.(p1 .+ 1e-30) - log.(p2 .+ 1e-30))
end
function DHa(theta1, theta2, calpha)
    return calpha' * (
        (theta1 .+ 1) / 2 .* log.((theta1 .+ 1 .+ 1e-30) ./ (theta2 .+ 1 .+ 1e-30)) +
            (1 .- theta1) / 2 .* log.((1 .- theta1 .+ 1e-30) ./ (1 .- theta2 .+ 1e-30))
    )
end
# function f(p, prob::EOTProblem)
#     return dot(p, prob.W)
# end
function φ(x, prob::EOTProblem)
    return sum(logsumexp(-prob.W / prob.η .+ x[1:prob.N] .+ x[(prob.N+1):end]')) - dot(prob.b, x)
end
function φ(u::TA, v::TA, r::TA, c::TA, W::TM, η::R) where {TA,TM,R}
    return -η * sum(logsumexp(-(W) / η .- u .- v')) - η * dot(r, u) - η * dot(c, v)
end

function get_p(x, prob::EOTProblem)
    return softmax(-prob.W / prob.η .+ x[1:prob.N] .+ x[(prob.N+1):end]')
end

function read_args_json(fpath::String; dtype::Type=Float64)
    json_string = read(fpath, String)
    settings = JSON3.read(json_string, EOTArgs{dtype})
    return settings
end
function softmax(x::AbstractArray{T}; normalize_values=true, dims=[], norm_dims=Nothing) where T<:Real
    if norm_dims == Nothing
        norm_dims = [1:ndims(x)...]
    end

    if !normalize_values
        return sum(exp.(x), dims=dims)
    else
        maxx = maximum(x, dims=norm_dims)
        v1 = sum(exp.(x .- maxx), dims=dims)
        return v1 ./ sum(v1, dims=norm_dims)
    end
end

function polyroot(a, b, c, γ)
    # Basic Newton's method routine to find the root of a polynomial
    dx = 1.
    if γ == 2
        return (-b + sqrt(b^2 - 4 * a * c)) / 2a
    end
    if γ == 1
        return -c / (a + b)
    end
    x = 1.0
    fx = a * x^(γ) + b * x + c
    niter_inner = 1
    tol = log2(a) - 20
    while log2(fx) > tol
        fx = a * x^(γ) + b * x + c
        dx = a * γ * x^(γ - 1) + b
        x = x - fx / dx
        # println(x, fx)
        niter_inner += 1
    end
    return x
end
function generate_random_ot(N, M, rng; dtype::Type=Float64)
    @assert dtype <: Real
    r = normalize(rand(rng, dtype, M), 1)
    c = normalize(rand(rng, dtype, N), 1)
    W = abs.(randn(rng, dtype, M, N))
    optimum = emd2(r, c, W)
    return r, c, W, optimum
end
function neg_entropy(x::AbstractArray{R}; dims=[]) where {R<:Real}
    return sum(map(y -> if y > 0.0
        y * log(y)
    else
        R(0.0)
    end, x), dims=dims)
end

function get_euclidean_distance(height::Int, width::Int; p::Float64=2.0, dtype::Type=Float64)
    @assert dtype <: Real
    N = height * width
    W = zeros(dtype, N, N)
    for (i, j) in product(0:(N-1), 0:(N-1))
        if p < 10
            W[i+1, j+1] = (abs(i ÷ height - j ÷ height)^p + abs(i % height - j % height)^p)
        else
            W[i+1, j+1] = max(abs(i ÷ height - j ÷ height), abs(i % height - j % height))
        end
    end
    return W
end
function round(γ::AbstractMatrix{T}, μ::AbstractArray{T}, ν::AbstractArray{T}) where T<:Real
    γ⁺ = γ .* min.(μ ./ (sum(γ, dims=2)), 1.0)
    γ⁺⁺ = γ⁺ .* min.(ν ./ (sum(γ⁺, dims=1))', 1.0)'
    rμ = μ - sum(γ⁺⁺, dims=2)
    rν = ν - sum(γ⁺⁺, dims=1)'
    γ̂ = γ⁺⁺ + rμ * rν' / norm(rμ, 1)
    return γ̂
end
function read_dotmark_data(fpath::String, sizes::Tuple{Int,Int}; dtype::Type=Float64)
    @assert dtype <: Real
    input_data = Matrix(dtype, CSV.read(fpath, header=false, DataFrame))
    # println(size(input_data), sizes)
    h = min(sizes[1], size(input_data, 1))
    w = min(sizes[2], size(input_data, 2))
    input_data = imresize(input_data, (h, w))
    N = h * w
    marginal = reshape(input_data, N) / sum(input_data)
    return marginal, h, w, N
end

function read_weights(fpath::String; dtype::Type=Float64)
    @assert dtype <: Real
    W = dtype.(Matrix(CSV.read(fpath, header=false, DataFrame)))
    return W
end