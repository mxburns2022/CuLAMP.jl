function max_logsumexp_cellsim!(
    output::CuDeviceVector{T},
    data::CuDeviceMatrix{T},
    row_sums::CuDeviceVector{T},
    row_sqnorms::CuDeviceVector{T},
    row_means::CuDeviceVector{T},
    row_centered_sqnorms::CuDeviceVector{T},
    scale::T,
    metric::Symbol,
) where T
    step = warpsize()
    nwarps = (gridDim().x * blockDim().x) ÷ step
    tid_x = (threadIdx().x + (blockIdx().x - 1) * blockDim().x - 1) ÷ step + 1
    N = size(data, 1)
    ncols = size(data, 2)
    N_outer = Int(ceil(N / nwarps))
    local_id = (threadIdx().x - 1) % step
    Ntiles = N ÷ step
    for _ in 1:N_outer
        if tid_x > N
            return
        end
        m_local = zero(T)
        for tile in 0:(Ntiles-1)
            j = tile * step + 1
            @inbounds costval = feature_cost_cuda(metric, data, row_sums, row_sqnorms, row_means, row_centered_sqnorms, scale, tid_x, j + local_id, ncols)
            m_local = max(m_local, costval)
        end
        if Ntiles * step + local_id < N
            j = Ntiles * step + 1
            @inbounds costval = feature_cost_cuda(metric, data, row_sums, row_sqnorms, row_means, row_centered_sqnorms, scale, tid_x, j + local_id, ncols)
            m_local = max(m_local, costval)
        end
        m_local = CUDA.CUDACore.reduce_warp(max, m_local)
        if local_id == 0
            output[tid_x] = m_local
        end
        tid_x += nwarps
    end
    return
end

function warp_min_reduce_cellsim!(
    output::CuDeviceVector{T},
    data::CuDeviceMatrix{T},
    row_sums::CuDeviceVector{T},
    row_sqnorms::CuDeviceVector{T},
    row_means::CuDeviceVector{T},
    row_centered_sqnorms::CuDeviceVector{T},
    scale::T,
    metric::Symbol,
    θ::CuDeviceVector{T},
    W∞::T,
) where T
    step = warpsize()
    nwarps = (gridDim().x * blockDim().x) ÷ step
    tid_x = (threadIdx().x + (blockIdx().x - 1) * blockDim().x - 1) ÷ step + 1
    N = size(data, 1)
    ncols = size(data, 2)
    N_outer = Int(ceil(N / nwarps))
    local_id = (threadIdx().x - 1) % step
    Ntiles = N ÷ step
    c1 = 2W∞
    for _ in 1:N_outer
        if tid_x > N
            return
        end
        m_local = T(Inf)
        for tile in 0:(Ntiles-1)
            j = tile * step + 1
            @inbounds begin
                muval = θ[j+local_id]
                costval = feature_cost_cuda(metric, data, row_sums, row_sqnorms, row_means, row_centered_sqnorms, scale, tid_x, j + local_id, ncols)
            end
            m_local = min(m_local, muladd(muval, c1, costval))
        end
        if Ntiles * step + local_id < N
            j = Ntiles * step + 1
            @inbounds begin
                muval = θ[j+local_id]
                costval = feature_cost_cuda(metric, data, row_sums, row_sqnorms, row_means, row_centered_sqnorms, scale, tid_x, j + local_id, ncols)
            end
            m_local = min(m_local, muladd(muval, c1, costval))
        end
        m_local = CUDA.CUDACore.reduce_warp(min, m_local)
        if local_id == 0
            output[tid_x] = m_local
        end
        tid_x += nwarps
    end
    return
end

function extragradient_cellsim(
    data::CuMatrix{T},
    scale::T,
    metric::Symbol,
    marginal1::CuArray{T},
    marginal2::CuArray{T},
    args::EOTArgs,
    frequency::Int=100,
    log_output::Bool=false
) where T<:Real
    N = M = size(data, 1)
    θ = CUDA.zeros(T, M)
    ν = copy(θ)
    θ̄ = copy(θ)
    ν̄ = copy(θ)
    row_sums, row_sqnorms, row_means, row_centered_sqnorms = get_flat_metrics(data)
    residual_cache = CUDA.zeros(T, M)
    sumvals = CUDA.zeros(T, N)
    threads = 256
    warp_blocks = div(N, div(threads, 32, RoundDown), RoundUp)
    linear_blocks = div(N, threads, RoundUp)
    @cuda threads = threads blocks = warp_blocks max_logsumexp_cellsim!(sumvals, data, row_sums, row_sqnorms, row_means, row_centered_sqnorms, scale, metric)
    CUDA.synchronize()
    W∞ = maximum(sumvals)
    η = T(args.eta_p)

    eta_mu = (marginal2 .+ T(args.alpha) / N) / args.tau_mu
    time_start = time_ns()
    st = T(0.0)
    cost_cache = CUDA.zeros(T, N)
    @inline function infeas(θt, ηt_value, s_value)
        @cuda threads = threads blocks = warp_blocks warp_logsumexp_spp_sim_fused!(sumvals, data, row_sums, row_sqnorms, row_means, row_centered_sqnorms, scale, metric, θt, ηt_value, s_value, W∞)
        CUDA.synchronize()
        @cuda threads = threads blocks = warp_blocks residual_cellsim_c!(residual_cache, cost_cache, data, row_sums, row_sqnorms, row_means, row_centered_sqnorms, scale, metric, marginal1, θt, sumvals, ηt_value, s_value, W∞)
        CUDA.synchronize()
    end
    hr = η * sum(neg_entropy(marginal1))
    println("time(s),iter,infeas,ot_objective,primal,dual,solver")
    W∞_scaling = one(T)

    st = one(T)
    ηt = T(Inf)

    τp = one(T)
    minv = tanh(-args.B / 2)
    maxv = tanh(args.B / 2)
    γ = copy(θ)
    for i in 1:args.itermax
        elapsed_time = (time_ns() - time_start) / 1e9
        if elapsed_time > args.tmax
            break
        end
        if args.verbose && (i - 1) % frequency == 0
            infeas(ν, ηt, st)
            CUDA.synchronize()
            residual_value = sum(abs.(residual_cache - marginal2))
            objective = sum(cost_cache)
            if η > 0
                primal_value = objective * (1 - η / ηt) - η * dot(marginal1, sumvals) - 2W∞ * η / ηt * dot(marginal2, ν) + hr + 2W∞ * residual_value
                @cuda threads = threads blocks = warp_blocks warp_logsumexp_spp_sim_fused!(sumvals, data, row_sums, row_sqnorms, row_means, row_centered_sqnorms, scale, metric, θ, η, one(T), W∞)
                dual_value = -η * dot(marginal1, sumvals) - 2W∞ * dot(marginal2, θ) + hr
            else
                primal_value = objective + 2W∞ * residual_value
                @cuda threads = threads blocks = warp_blocks warp_min_reduce_cellsim!(sumvals, data, row_sums, row_sqnorms, row_means, row_centered_sqnorms, scale, metric, γ, W∞)
                CUDA.synchronize()
                dual_value = dot(marginal1, sumvals) - 2W∞ * dot(marginal2, γ)
            end

            CUDA.synchronize()

            @printf "%.6e,%d,%.14e,%.14e,%.14e,%.14e,lamp_kernel\n" elapsed_time i residual_value objective primal_value dual_value
            if primal_value - dual_value < args.epsilon / 6 && residual_value < args.epsilon / 6
                break
            end
        end
        infeas(ν, ηt, st)
        @cuda threads = threads blocks = linear_blocks update_θ_residual(θ̄, θ, residual_cache, marginal2, eta_mu, T(args.eta_mu), false, minv, maxv, one(T))
        ηt = one(T) / (τp + (one(T) / ηt) * (one(T) - η))
        ν̄ .= (one(T) - ηt) * ν + ηt * θ

        CUDA.synchronize()
        st = (one(T) - ηt) * st + ηt
        infeas(ν̄, ηt, st)
        @cuda threads = threads blocks = linear_blocks update_θ_residual(θ, θ, residual_cache, marginal2, eta_mu, T(args.eta_mu), true, minv, maxv, one(T))
        ν .= (one(T) - ηt) * ν + ηt * θ̄
        γ .= (one(T) - ηt) * γ + ηt * θ

        CUDA.synchronize()
    end

    ψ = -2W∞ * ν ./ args.eta_p
    φ = log.(marginal1) - sumvals
    objective = sum(cost_cache)
    return Array(ν), Array(φ), Array(ψ), objective
end

function residual_cellsim_c!(
    output::CuDeviceVector{T},
    cost_output::CuDeviceVector{T},
    data::CuDeviceMatrix{T},
    row_sums::CuDeviceVector{T},
    row_sqnorms::CuDeviceVector{T},
    row_means::CuDeviceVector{T},
    row_centered_sqnorms::CuDeviceVector{T},
    scale::T,
    metric::Symbol,
    marginal1::CuDeviceVector{T},
    θ::CuDeviceVector{T},
    logZi::CuDeviceVector{T},
    reg::T,
    st::T,
    W∞::T,
) where T
    step = warpsize()
    nwarps = (gridDim().x * blockDim().x) ÷ step
    tid_x = (threadIdx().x + (blockIdx().x - 1) * blockDim().x - 1) ÷ step + 1
    N = M = size(data, 1)
    ncols = size(data, 2)
    N_outer = Int(ceil(M / nwarps))
    local_id = (threadIdx().x - 1) % step
    c1 = st / 2W∞
    invreg = one(T) / reg
    Ntiles = N ÷ step
    for _ in 1:N_outer
        local_acc = zero(T)
        cost_acc = zero(T)
        if tid_x > M
            continue
        end
        @inbounds diff = θ[tid_x]
        for tile in 0:(Ntiles-1)
            j = tile * step + 1
            @inbounds begin
                costval = feature_cost_cuda(metric, data, row_sums, row_sqnorms, row_means, row_centered_sqnorms, scale, j + local_id, tid_x, ncols)
                marginalv = marginal1[j+local_id]
            end
            value = muladd(muladd(costval, c1, diff), -invreg, -logZi[j+local_id])
            weight = exp(value) * marginalv
            local_acc += weight
            cost_acc += weight * costval
        end
        if Ntiles * step + local_id < N
            j = Ntiles * step + 1
            @inbounds begin
                costval = feature_cost_cuda(metric, data, row_sums, row_sqnorms, row_means, row_centered_sqnorms, scale, j + local_id, tid_x, ncols)
                marginalv = marginal1[j+local_id]
            end
            value = muladd(muladd(costval, c1, diff), -invreg, -logZi[j+local_id])
            weight = exp(value) * marginalv
            local_acc += weight
            cost_acc += weight * costval
        end
        local_acc = CUDA.CUDACore.reduce_warp(+, local_acc)
        cost_acc = CUDA.CUDACore.reduce_warp(+, cost_acc)
        if local_id == 0
            @inbounds begin
                output[tid_x] = local_acc
                cost_output[tid_x] = cost_acc
            end
        end
        tid_x += nwarps
    end
    return
end

function warp_logsumexp_spp_sim_fused!(
    output::CuDeviceVector{T},
    data::CuDeviceMatrix{T},
    row_sums::CuDeviceVector{T},
    row_sqnorms::CuDeviceVector{T},
    row_means::CuDeviceVector{T},
    row_centered_sqnorms::CuDeviceVector{T},
    scale::T,
    metric::Symbol,
    θ::CuDeviceVector{T},
    reg::T,
    st::T,
    W∞::T,
) where T
    step = warpsize()
    nwarps = (gridDim().x * blockDim().x) ÷ step
    tid_x = (threadIdx().x + (blockIdx().x - 1) * blockDim().x - 1) ÷ step + 1
    N = M = size(data, 1)
    ncols = size(data, 2)

    N_outer = Int(ceil(M / nwarps))
    local_id = (threadIdx().x - 1) % step
    c1 = st / 2W∞
    invreg = one(T) / reg
    Ntiles = M ÷ step
    for _ in 1:N_outer
        if tid_x > N
            return
        end

        m_local = T(-Inf)
        s_local = zero(T)

        for tile in 0:(Ntiles-1)
            j = tile * step + 1
            @inbounds begin
                muval = θ[j+local_id]
                costval = feature_cost_cuda(metric, data, row_sums, row_sqnorms, row_means, row_centered_sqnorms, scale, tid_x, j + local_id, ncols)
            end
            v = -(muladd(costval, c1, muval)) * invreg
            if v <= m_local
                s_local += exp(v - m_local)
            else
                s_local = s_local * exp(m_local - v) + one(T)
                m_local = v
            end
        end
        if Ntiles * step + local_id < M
            j = Ntiles * step + 1
            @inbounds begin
                muval = θ[j+local_id]
                costval = feature_cost_cuda(metric, data, row_sums, row_sqnorms, row_means, row_centered_sqnorms, scale, tid_x, j + local_id, ncols)
            end
            v = -(muladd(costval, c1, muval)) * invreg
            if v <= m_local
                s_local += exp(v - m_local)
            else
                s_local = s_local * exp(m_local - v) + one(T)
                m_local = v
            end
        end
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
