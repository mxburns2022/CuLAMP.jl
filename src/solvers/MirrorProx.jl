using CUDA
using Printf
using LinearAlgebra
using Random
using Test

function dualv(theta, W, W∞, ηp, r, c)
    if ηp == 0
        return sum(-dot(c, 2W∞ .* (theta))) + sum(r' * minimum((W .+ 2W∞ .* theta'), dims=2))
    end
    lse = logsumexp(-(0.5 * W .+ (W∞ .* theta)') ./ ηp, 2)
    return 2 * ((-W∞ * sum(c' * theta) - ηp * sum(r' * lse)) + ηp * dot(r, log.(r)))
end
function primalv(θ::TA, W::TM, W∞::TWinf, ηt::R, ηp::R, r::TA, c::TA) where {TA,TM,R,TWinf}
    pμ = softmax(-(0.5W .+ (W∞ .* θ)') ./ ηt, norm_dims=2)
    lse = logsumexp(-(0.5 * W .+ (W∞ .* θ)') ./ ηt, 2)
    primal_other = (0.5 * (1 - ηp / ηt)dot(W, r .* pμ) + sum(W∞ .* abs.(sum(r .* pμ, dims=1)' - c)) + (-W∞ * ηp / ηt * sum(c' * θ) - ηp * sum(r' * lse)) + ηp * dot(r, log.(r)))
    primal_val = (0.5 * dot(W, r .* pμ)
        + sum(W∞ .* abs.(sum(r .* pμ, dims=1)' - c))
        + ηp * dot(r, neg_entropy(pμ, dims=2)) + ηp * dot(r, log.(r)))
    # println(primal_val, " ", primal_other)
    # sleep(1)
    return 2(0.5 * dot(W, r .* pμ)
        + sum(W∞ .* abs.(sum(r .* pμ, dims=1)' - c))
        + ηp * dot(r, neg_entropy(pμ, dims=2)) + ηp * dot(r, log.(r)))
end

function primalv(p::TM, W::TM, W∞::TWinf, ηp::R, r::TA, c::TA) where {TA,TM,R,TWinf}
    return 2(0.5 * dot(W, r .* p) +
        sum(W∞ .* abs.(sum(r .* p, dims=1)' - c)) +
        ηp * dot(r, neg_entropy(p, dims=2)) + ηp * dot(r, log.(r)))
end
function PDMP(r::AbstractArray{R},
    c::AbstractArray{R},
    W::TW,
    args::EOTArgs{R},
    frequency::Int=50;
    adjust::Bool=true,
    log_output=false,
    p0=Nothing
) where {R,TW}
    # input 
    # W∞ = maximum(W', dims=2)
    W∞ = maximum(W)
    ηp = args.eta_p / 2
    n = size(r, 1)
    if isa(W, CuArray)
        μ⁺ = 0.5 * CUDA.ones(R, (n, 1))
    else
        μ⁺ = 0.5 * ones(R, (n, 1))
    end

    μ⁻ = copy(μ⁺)
    ν⁺ = copy(μ⁺)
    ν⁻ = copy(μ⁺)
    μ⁻t = copy(μ⁺)
    μ⁺t = copy(μ⁺)
    # eta_mu = args.C2 * sqrt(args.B) ./ (c .+ args.C3)
    eta_mu = 1. ./ (c .+ args.alpha / n) ./ args.tau_mu
    ηπ = 1 ./ r ./ args.tau_p
    if p0 == Nothing
        if isa(W, CuArray)
            p = CUDA.ones(R, (n, n)) ./ (n)
        else
            p = ones(R, (n, n)) ./ (n)
        end
    else
        if isa(W, CuArray)
            p = CuArray(p0) ./ r
        else
            p = (p0) ./ r
        end
    end

    println("time(s),iter,infeas,ot_objective,primal,dual,solver")
    time_start = time_ns()
    ν = copy(μ⁺)
    ν⁻ = copy(μ⁺)
    calpha = (c .+ args.alpha / n)

    sumv = 0.
    val1 = 0
    val2 = 0
    loglist = []
    for i in 1:args.itermax
        elapsed_time = (time_ns() - time_start) / 1e9
        if elapsed_time > args.tmax
            break
        end
        # ηp = 1/(max(i-1, 1))
        # eta_mu = 1 ./ sum(r .* p, dims=1)'
        arg = (sum(r .* p, dims=1)' - c) .* eta_mu
        maxval = max.(arg, -arg)
        μ⁻t = μ⁻ .^ (1 - args.tau_mu * args.eta_mu) .* exp.(-arg - maxval)
        μ⁺t = μ⁺ .^ (1 - args.tau_mu * args.eta_mu) .* exp.(arg - maxval)
        normv = (μ⁻t + μ⁺t)
        μ⁻t = μ⁻t ./ normv
        μ⁺t = μ⁺t ./ normv

        # sleep(0.1)
        pt = p .^ (1 - args.tau_p * ηp) .* exp.(-(ηπ .* r) .* (W * 0.5 / W∞ .+ (μ⁺ .- μ⁻)'))
        pt ./= sum(pt, dims=2)
        if log_output
            val1 = DHa(μ⁺t - μ⁻t, μ⁺ - μ⁻, calpha) + KL(pt, p, r) * 0.5 / W∞
        end

        # eta_mu = 1 ./ sum(r .* pt, dims=1)'
        # println(norm(pt - ptest))
        arg = (sum(r .* pt, dims=1)' - c) .* eta_mu
        maxval = max.(arg, -arg)
        μ⁻ = μ⁻ .^ (1 - args.tau_mu * args.eta_mu) .* exp.(-arg - maxval)
        μ⁺ = μ⁺ .^ (1 - args.tau_mu * args.eta_mu) .* exp.(arg - maxval)
        # println(size(μ⁺))
        normv = (μ⁻ + μ⁺)
        μ⁻ ./= normv
        μ⁺ ./= normv
        p = p .^ (1 - args.tau_p * ηp) .* exp.(-(ηπ .* r) .* (W * 0.5 / W∞ .+ (μ⁺t .- μ⁻t)'))
        p ./= sum(p, dims=2)
        if log_output
            val2 = DHa(μ⁺t - μ⁻t, μ⁺ - μ⁻, calpha) + KL(pt, p, r) * 0.5 / W∞
        end
        if (i - 1) % frequency == 0
            # display(p)
            pr = r .* p
            feas = norm(sum(pr', dims=2) - c, 1)
            obj = dot(pr, W)
            pobj = primalv(p, W, W∞, ηp, r, c)
            dobj = dualv(μ⁺ - μ⁻, W, W∞, ηp, r, c)
            if args.verbose
                @printf "%.6e,%d,%.14e,%.14e,%.14e,%.14e,pdmp\n" elapsed_time i feas obj pobj dobj
            end
            if pobj - dobj < args.epsilon / 6 && feas < args.epsilon / 6
                break
            end
        end
        if log_output
            push!(loglist, (cross_term_1=val1, cross_term_2=val2))
        end
        # println(norm(ptest-p), " ", ηp)
        # sleep(1)
        # p = p .^ (1 - ηp) .* exp.(-(ηπ .* r) .* (W * 0.5 / W∞ .+ (μ⁺t .- μ⁻t)'))
        if adjust
            μ⁻a = max.(μ⁻, exp(-args.B) .* max.(μ⁺, μ⁻))
            μ⁺a = max.(μ⁺, exp(-args.B) .* max.(μ⁺, μ⁻))
            normv = (μ⁻a + μ⁺a)
            μ⁻ = μ⁻a ./ normv
            μ⁺ = μ⁺a ./ normv
        else
            copy!(μ⁺a, μ⁺)
            copy!(μ⁻a, μ⁻)
        end
    end
    if log_output
        return round(r .* p, r, c), μ⁺ .- μ⁻, loglist
    end
    return round(r .* p, r, c), μ⁺, μ⁻
end

function update_θ_residual(theta::CuDeviceArray{R}, theta_0::CuDeviceArray{R}, residual::CuDeviceArray{R}, c::CuDeviceArray{R}, eta_mu_i::CuDeviceArray{R}, eta_mu::R, adjust::Bool, minv::R, maxv::R, W∞::R) where R
    N = size(theta, 1)
    tid = threadIdx().x + blockDim().x * (blockIdx().x - 1)
    if tid > N
        return
    end
    @inbounds begin
        difference = 2W∞ * (residual[tid] - c[tid]) / eta_mu_i[tid]
        thetav = theta_0[tid]
    end
    maxval = max(difference, -difference)
    expv = exp(difference - maxval) * ((thetav + 1) / (1 - thetav)) .^ (1 - eta_mu)
    theta_value_new = (expv - exp(-maxval)) / (expv + exp(-maxval))
    if adjust
        theta_value_new = clamp(theta_value_new, minv, maxv)#max(min(theta_value_new, maxv), minv)
    end
    @inbounds begin
        theta[tid] = theta_value_new
    end

    return
end

function warp_logsumexp!(output::CuDeviceVector{T}, W::CuDeviceMatrix{T}, θ::CuDeviceVector{T}, reg::T, st::T, W∞::T) where T
    step = warpsize()
    nwarps = (gridDim().x * blockDim().x) ÷ step
    tid_x = (threadIdx().x + (blockIdx().x - 1) * blockDim().x - 1) ÷ step + 1
    N = size(W, 1)
    N_outer = Int(ceil(N / nwarps))
    local_id = (threadIdx().x - 1) % step
    for i in 1:N_outer
        if tid_x > N
            return
        end
        maxval = -Inf
        for i in 1:step:N
            if i + local_id > N
                break
            end
            value = -(0.5 * W[i+local_id, tid_x] / W∞ * st + (θ[i+local_id])) / reg
            maxval = max(value, maxval)
        end
        maxval = CUDACore.reduce_warp(max, maxval)
        sync_warp()
        if local_id == 0
            output[tid_x] = maxval
        end
        # return
        maxval = output[tid_x]

        local_acc = 0.0
        for i in 1:step:N
            if i + local_id > N
                break
            end
            value = -(0.5 * W[i+local_id, tid_x] / W∞ * st + θ[i+local_id]) / reg
            local_acc += exp(value - maxval)
        end
        local_acc2 = CUDACore.reduce_warp(+, local_acc)
        if local_id == 0
            output[tid_x] = (log(local_acc2) + maxval)
        end
        tid_x += nwarps
    end
    return
end


# Fused, coalesced variant: expects Wᵗ (i.e., permutedims(W, (2,1))) so that loads across lanes are coalesced.
# Performs a single pass using an associative log-sum-exp pair reduction to avoid re-reading memory.
@inline function _lse_pair_combine(a::NTuple{2,T}, b::NTuple{2,T}) where {T}
    m1, s1 = a
    m2, s2 = b
    m = ifelse(m1 > m2, m1, m2)
    # s = s1*exp(m1-m) + s2*exp(m2-m)
    return (m, s1 * exp(m1 - m) + s2 * exp(m2 - m))
end

function warp_logsumexp_fused!(output::CuDeviceVector{T}, Wt::CuDeviceMatrix{T}, θ::CuDeviceVector{T}, reg::T, st::T, Winf::T) where T
    step = warpsize()
    nwarps = (gridDim().x * blockDim().x) ÷ step
    tid_x = (threadIdx().x + (blockIdx().x - 1) * blockDim().x - 1) ÷ step + 1
    N = size(Wt, 1)  # Wt is transposed: first dim iterates columns j
    M = size(Wt, 2)
    N_outer = Int(ceil(M / nwarps))
    local_id = (threadIdx().x - 1) % step
    # Precompute scalars to avoid divisions in the inner loop
    α = (st) / 2(reg * Winf)
    invreg = one(T) / reg
    @inbounds for _ in 1:N_outer
        if tid_x > M
            return
        end
        # Online log-sum-exp per thread
        m_local = T(-Inf)
        s_local = zero(T)
        @inbounds for j in 1:step:N
            jj = j + local_id
            if jj > N
                break
            end
            # Access W as W[jj, tid_x] because we pass Wᵀ for coalesced loads
            w = Wt[jj, tid_x]
            δ = θ[jj] * invreg
            v = -(α * w + δ)
            # online LSE update
            if v <= m_local
                s_local += exp(v - m_local)
            else
                s_local = s_local * exp(m_local - v) + one(T)
                m_local = v
            end
        end

        # Warp-level reduction of (m,s)
        m = shfl_down_sync(0xffffffff, m_local, 16)
        s = shfl_down_sync(0xffffffff, s_local, 16)
        m_local, s_local = _lse_pair_combine((m_local, s_local), (m, s))
        m = shfl_down_sync(0xffffffff, m_local, 8)
        s = shfl_down_sync(0xffffffff, s_local, 8)
        m_local, s_local = _lse_pair_combine((m_local, s_local), (m, s))
        m = shfl_down_sync(0xffffffff, m_local, 4)
        s = shfl_down_sync(0xffffffff, s_local, 4)
        m_local, s_local = _lse_pair_combine((m_local, s_local), (m, s))
        m = shfl_down_sync(0xffffffff, m_local, 2)
        s = shfl_down_sync(0xffffffff, s_local, 2)
        m_local, s_local = _lse_pair_combine((m_local, s_local), (m, s))
        m = shfl_down_sync(0xffffffff, m_local, 1)
        s = shfl_down_sync(0xffffffff, s_local, 1)
        m, s = _lse_pair_combine((m_local, s_local), (m, s))

        if local_id == 0
            output[tid_x] = log(s) + m
        end
        tid_x += nwarps
    end
    return
end


function residual_c!(output::CuDeviceVector{T}, r::CuDeviceArray{T}, W::CuDeviceMatrix{T}, θ::CuDeviceVector{T}, logZi::CuDeviceVector{T}, reg::T, st::T, Winf::T) where T
    step = warpsize()
    nwarps = (gridDim().x * blockDim().x) ÷ step
    tid_x = (threadIdx().x + (blockIdx().x - 1) * blockDim().x - 1) ÷ step + 1
    M = size(r, 1)
    N = size(θ, 1)
    N_outer = Int(ceil(N / nwarps))
    local_id = (threadIdx().x - 1) % step
    for i in 1:N_outer
        if tid_x > N
            return
        end
        @inbounds begin
            diff = (θ[tid_x])
        end
        local_acc = 0.0
        for i in 1:step:M
            if i + local_id > M
                break
            end
            @inbounds begin
                value = -(0.5 * st * W[i+local_id, tid_x] ./ Winf + diff) / reg - logZi[i+local_id]
                local_acc += r[i+local_id] * exp(value)
            end
        end
        local_acc2 = CUDACore.reduce_warp(+, local_acc)
        if local_id == 0
            @inbounds begin
                output[tid_x] = local_acc2
            end
        end
        tid_x += nwarps
    end
    return
end

function test_warp_logsumexp!()
    rng = Xoshiro(0)
    N = 20
    r = CuArray(normalize!(rand(rng, N), 1))
    c = CuArray(normalize!(rand(rng, N), 1))
    μ⁺ = CuArray(rand(rng, N))
    μ⁻ = CuArray(rand(rng, N))
    maxvals = CUDA.zeros(Float64, N)
    sumvals = CUDA.zeros(Float64, N)
    residual_storage = CUDA.zeros(Float64, N)
    residual_storage2 = CUDA.zeros(Float64, N)
    normv = (μ⁺ + μ⁻)
    μ⁺ ./= normv
    μ⁻ ./= normv

    W = CuArray(rand(rng, N, N))
    ηp = 1e-3
    st = 0.02
    maximum!(maxvals, -(W * 0.5 * st .+ μ⁺' .- μ⁻') ./ ηp)
    threads = 256
    warp_blocks = div(N, div(threads, 32, RoundUp), RoundUp)
    sum!(sumvals, (exp.(-(W * 0.5 * st .+ μ⁺' .- μ⁻') ./ ηp .- maxvals)))
    sumvals = log.(sumvals) + maxvals
    # sumvals = maxvals
    sumvals2 = CUDA.zeros(Float64, N)
    # Use transposed W for coalesced memory access in the fused kernel
    Wt = permutedims(W, (2, 1))
    @cuda threads = threads blocks = warp_blocks warp_logsumexp_fused!(sumvals2, Wt, θ, ηp, st, W∞)
    CUDA.synchronize()
    println(sumvals)
    println(sumvals2)
    @test norm(sumvals - sumvals2) ≈ 0 atol = 1e-8

    sum!(residual_storage', exp.(-(W * 0.5 * st .+ μ⁺' .- μ⁻') ./ ηp .- sumvals .+ log.(r)))
    # @test sum(residual_storage) ≈ 1.0 atol = 1e-8
    @cuda threads = threads blocks = warp_blocks residual_c!(residual_storage2, r, W, μ⁺, μ⁻, sumvals, ηp, st)
    CUDA.synchronize()
    println(residual_storage2)
    println(residual_storage)
    @test norm(residual_storage2 - residual_storage) ≈ 0 atol = 1e-8

    println(CUDA.@elapsed begin
        maximum!(maxvals, -(W * 0.5 * st .+ μ⁺' .- μ⁻') ./ ηp)
        sum!(sumvals, (exp.(-(W * 0.5 * st .+ μ⁺' .- μ⁻') ./ ηp .- maxvals)))
        sum!(residual_storage', exp.(-(W * 0.5 * st .+ μ⁺' .- μ⁻') ./ ηp .- maxvals .- log.(sumvals) .+ log.(r)))
    end)

    println(CUDA.@elapsed begin
        @cuda threads = threads blocks = warp_blocks warp_logsumexp_fused!(sumvals2, Wt, μ⁺, ηp, st, W∞)
        @cuda threads = threads blocks = warp_blocks residual_c!(residual_storage2, r, W, μ⁺, μ⁻, sumvals, ηp, st)
    end)


end



# +-------------------------+
# |   This one sparks joy   |
# +-------------------------+
#       (◠‿◠)
#     ノ( ♥ )
function LAMP(r::CuArray{R},
    c::CuArray{R},
    W::CuArray{R},
    args::EOTArgs{R},
    frequency::Int=50;
    log_output::Bool=false,
    _θ=Nothing
) where {R}
    W∞ = norm(W, Inf)
    # ε /= W∞
    n = size(c, 1)
    m = size(r, 1)
    @assert size(W) == (m, n)
    θ = if _θ == Nothing
        CUDA.zeros(R, n)
    else
        copy(_θ)
    end
    ηp = args.eta_p / 2 / W∞
    ηstep = ηp * W∞
    sumvals = CUDA.zeros(R, m)
    residual_storage = CUDA.zeros(R, (n))
    ν = copy(θ)
    ν̄ = copy(θ)
    θ̄ = copy(θ)
    eta_mu = (c .+ args.alpha / n) ./ (args.tau_mu)
    threads = 256
    if args.verbose
        println("time(s),iter,infeas,ot_objective,primal,dual,pdgap,solver")
    end
    time_start = time_ns()
    linear_blocks = Int(ceil(n / threads))
    warp_blocks = Int(ceil(n / div(threads, 32, RoundUp)))
    # Precompute a transposed copy for coalesced reads in the fused LSE kernel
    Wt = permutedims(W, (2, 1))
    @inline function infeas(θ, η, s)
        @cuda threads = threads blocks = warp_blocks warp_logsumexp_fused!(sumvals, Wt, θ, η, s, W∞)
        @cuda threads = threads blocks = warp_blocks residual_c!(residual_storage, r, W, θ, sumvals, η, s, W∞)
    end
    # ηp = 0.1
    ηt = R(Inf)
    τp = args.tau_p
    minv = tanh(-args.B / 2)
    maxv = tanh(args.B / 2)
    onev = R(1.0)
    infeas(ν, ηt, onev)
    calpha = (c .+ args.alpha / n)

    loglist = []
    sumv = 0.0
    val1 = 0
    val2 = 0
    for i in 1:args.itermax
        elapsed_time = (time_ns() - time_start) / 1e9
        if elapsed_time > args.tmax
            break
        end
        if (i - 1) % frequency == 0
            p = softmax(-(W / 2W∞ .+ ν') ./ ηt, norm_dims=2)
            obj = dot(W, round(r .* p, r, c))
            # pr = r .* p
            primal_value = primalv(p, W, W∞, ηp, r, c)
            dual_value = dualv(θ, W, W∞, ηp, r, c)
            # feas = norm(sum(pr, dims=1)' - c, 1)
            feas = norm(c - sum((r .* p)', dims=2), 1)
            if args.verbose
                @printf "%.6e,%d,%.14e,%.14e,%.14e,%.14e,%.14e,lamp_cuda\n" elapsed_time i feas obj primal_value dual_value primal_value - dual_value
            end
            if feas < args.epsilon / 2 && primal_value - dual_value < args.epsilon / 2
                break
            end
        end
        @cuda threads = threads blocks = linear_blocks update_θ_residual(θ̄, θ, residual_storage, c, eta_mu, args.tau_mu * args.eta_mu, false, minv, maxv, onev)
        eta_prev = ηt
        ηt = 1 / (τp + (1 / ηt) * (1 - ηp))

        ν̄ .= τp * ηt .* θ + (1 - τp * ηt) .* ν
        if log_output
            val1 = DHa(θ̄, θ, calpha) + KL(ν̄, ν, ηt, eta_prev, r, W, W∞)
        end

        # if args.eta_p == 0
        #     ηp = 1 / (i)
        # end
        infeas(ν̄, ηt, onev)
        diffterm = dot(ν ./ ηt - ν̄ ./ ηt, c - residual_storage)
        @cuda threads = threads blocks = linear_blocks update_θ_residual(θ, θ, residual_storage, c, eta_mu, args.eta_mu, false, minv, maxv, onev)
        ν = (1 - τp * ηt) * ν + τp * ηt * θ̄
        if log_output
            val2 = DHa(θ̄, θ, calpha) + KL(ν̄, ν, ηt, ηt, r, W, W∞)
        end

        θ .= clamp.(θ, minv, maxv)

        diffterm = dot(ν - ν̄, c - residual_storage)
        infeas(ν, ηt, onev)
        if log_output
            push!(loglist, (iteration=i, cross_term_1=val1, cross_term_2=val2))
        end
    end
    p = softmax(-(W ./ 2W∞ .+ ν') ./ ηt, norm_dims=2)
    if log_output
        return r .* p, θ, loglist
    end
    return r .* p, θ

end


function LAMP(r::AbstractArray{R},
    c::AbstractArray{R},
    W::AbstractMatrix{R},
    args::EOTArgs{R},
    frequency::Int=50;
    s0::Float64=0.0
) where {R}
    W∞ = norm(W, Inf)
    ηp = args.eta_p / 2W∞
    n = size(r, 1)

    θ = zeros(R, n)
    ν = zeros(R, n)
    θ̄ = zeros(R, n)
    ν̄ = zeros(R, n)

    sumvals = zeros(R, n)
    residual_storage = zeros(R, n)
    maxvals = zeros(R, n)

    eta_mu = (c .+ args.alpha / n) ./ (args.tau_mu)
    st = 0
    ηt = R(Inf)
    τp = args.tau_p
    println("time(s),iter,infeas,ot_objective,primal,dual,solver")
    time_start = time_ns()
    function infeas(θ, ηt)

        maximum!(maxvals, -(W / 2W∞ .+ θ') ./ ηt)
        sum!(sumvals, exp.(-(W / 2W∞ .+ θ') ./ ηt .- maxvals))
        sum!(residual_storage', exp.(-(W ./ 2W∞ .+ θ') ./ ηt .- maxvals .- log.(sumvals) .+ log.(r)))
        # return 
    end
    γ = zeros(R, n)
    for i in 1:args.itermax
        elapsed_time = (time_ns() - time_start) / 1e9
        if elapsed_time > args.tmax
            break
        end



        if (i - 1) % frequency == 0
            p = softmax(-(W / 2W∞ .+ ν') ./ ηt, norm_dims=2)
            pr = r .* p
            feas = norm(sum(pr, dims=1)' - c, 1)
            obj = dot(round(pr, r, c), W)
            pobj = primalv(p, W, W∞, ηp, r, c)
            dobj = dualv(γ, W, W∞, ηp, r, c)
            if args.verbose
                @printf "%.6e,%d,%.14e,%.14e,%.14e,%.14e,lamp\n" elapsed_time i feas obj pobj dobj
            end
            if (pobj - dobj) < args.epsilon / 6 && feas < args.epsilon / 6
                break
            end
        end

        infeas(ν, ηt)
        θ̄ .= tanh.((residual_storage - c) ./ eta_mu + 1 / 2 .* (1 .- args.tau_mu .* args.eta_mu) .* log.((θ .+ 1) ./ (1 .- θ)))
        ηt = 1 / (τp + (1 / ηt) * (1 - ηp))
        ν̄ .= ν + τp * ηt * (θ - ν)
        infeas(ν̄, ηt)
        θ .= tanh.((residual_storage - c) ./ eta_mu + 1 / 2 .* (1 .- args.tau_mu * args.eta_mu) .* log.((θ .+ 1) ./ (1 .- θ)))
        γ .= θ + τp * ηt * (θ - γ)
        clamp!(θ, tanh(-args.B / 2), tanh(args.B / 2))
        ν .= ν + τp * ηt * (θ̄ - ν)
    end
    p = softmax(-(W / 2W∞ .+ ν') ./ ηt, norm_dims=2)
    return r .* p, θ
end

