function residual_cellsim_opt!(
    output::CuDeviceVector{T},
    cost_output::CuDeviceVector{T},
    data1::CuDeviceMatrix{T},
    data2::CuDeviceMatrix{T},
    row_sums1::CuDeviceVector{T},
    row_sqnorms1::CuDeviceVector{T},
    row_means1::CuDeviceVector{T},
    row_centered_sqnorms1::CuDeviceVector{T},
    row_sums2::CuDeviceVector{T},
    row_sqnorms2::CuDeviceVector{T},
    row_means2::CuDeviceVector{T},
    row_centered_sqnorms2::CuDeviceVector{T},
    scale::T,
    metric::Symbol,
    marginal::CuDeviceVector{T},
    φ::CuDeviceVector{T},
    ψ::CuDeviceVector{T},
    reg::T,
    W∞::T,
) where T
    step = warpsize()
    nwarps = (gridDim().x * blockDim().x) ÷ step
    tid_x = (threadIdx().x + (blockIdx().x - 1) * blockDim().x - 1) ÷ step + 1
    N = size(data1, 1)
    M = size(data2, 1)
    ncols = size(data1, 2)
    N_outer = Int(ceil(N / nwarps))
    local_id = (threadIdx().x - 1) % step
    invreg = -one(T) / (reg * W∞)
    Ntiles = M ÷ step
    for _ in 1:N_outer
        if tid_x > N
            return
        end
        local_acc = zero(T)
        cost_acc = zero(T)
        @inbounds φi = φ[tid_x]
        for tile in 0:(Ntiles-1)
            j = tile * step + 1
            @inbounds begin
                muval = ψ[j+local_id]
                costval = feature_cost_cuda(metric, data1, row_sums1, row_sqnorms1, row_means1, row_centered_sqnorms1, scale, tid_x, j + local_id, ncols)
            end
            value = muladd(costval, invreg, φi) + muval
            weight = exp(value)
            local_acc += weight
            cost_acc += weight * costval
        end
        if Ntiles * step + local_id < M
            j = Ntiles * step + 1
            @inbounds begin
                muval = ψ[j+local_id]
                costval = feature_cost_cuda(metric, data1, row_sums1, row_sqnorms1, row_means1, row_centered_sqnorms1, scale, tid_x, j + local_id, ncols)
            end
            value = muladd(costval, invreg, φi) + muval
            weight = exp(value)
            local_acc += weight
            cost_acc += weight * costval
        end
        local_acc = CUDA.CUDACore.reduce_warp(+, local_acc)
        cost_acc = CUDA.CUDACore.reduce_warp(+, cost_acc)
        if local_id == 0
            @inbounds begin
                output[tid_x] = marginal[tid_x] - local_acc
                cost_output[tid_x] = cost_acc
            end
        end
        tid_x += nwarps
    end
    return
end

function warp_logsumexp_cellsim_opt!(
    output::CuDeviceVector{T},
    data1::CuDeviceMatrix{T},
    data2::CuDeviceMatrix{T},
    row_sums1::CuDeviceVector{T},
    row_sqnorms1::CuDeviceVector{T},
    row_means1::CuDeviceVector{T},
    row_centered_sqnorms1::CuDeviceVector{T},
    row_sums2::CuDeviceVector{T},
    row_sqnorms2::CuDeviceVector{T},
    row_means2::CuDeviceVector{T},
    row_centered_sqnorms2::CuDeviceVector{T},
    scale::T,
    metric::Symbol,
    marginal::CuDeviceVector{T},
    ψ::CuDeviceVector{T},
    reg::T,
    W∞::T,
) where T
    step = warpsize()
    nwarps = (gridDim().x * blockDim().x) ÷ step
    tid_x = (threadIdx().x + (blockIdx().x - 1) * blockDim().x - 1) ÷ step + 1
    N = size(data1, 1)
    M = size(data2, 1)
    ncols = size(data1, 2)
    N_outer = Int(ceil(N / nwarps))
    local_id = (threadIdx().x - 1) % step
    invreg = -one(T) / (reg * W∞)
    Ntiles = M ÷ step
    for _ in 1:N_outer
        if tid_x > N
            return
        end
        maxval = T(-Inf)
        for tile in 0:(Ntiles-1)
            j = tile * step + 1
            @inbounds begin
                muval = ψ[j+local_id]
                costval = feature_cost_cuda(metric, data1, row_sums1, row_sqnorms1, row_means1, row_centered_sqnorms1, scale, tid_x, j + local_id, ncols)
            end
            maxval = max(maxval, muladd(costval, invreg, muval))
        end
        if Ntiles * step + local_id < M
            j = Ntiles * step + 1
            @inbounds begin
                muval = ψ[j+local_id]
                costval = feature_cost_cuda(metric, data1, row_sums1, row_sqnorms1, row_means1, row_centered_sqnorms1, scale, tid_x, j + local_id, ncols)
            end
            maxval = max(maxval, muladd(costval, invreg, muval))
        end
        maxval = CUDA.CUDACore.reduce_warp(max, maxval)
        maxval = CUDA.shfl_sync(CUDA.FULL_MASK, maxval, 1)

        local_acc = zero(T)
        for tile in 0:(Ntiles-1)
            j = tile * step + 1
            @inbounds begin
                muval = ψ[j+local_id]
                costval = feature_cost_cuda(metric, data1, row_sums1, row_sqnorms1, row_means1, row_centered_sqnorms1, scale, tid_x, j + local_id, ncols)
            end
            local_acc += exp(muladd(costval, invreg, muval) - maxval)
        end
        if Ntiles * step + local_id < M
            j = Ntiles * step + 1
            @inbounds begin
                muval = ψ[j+local_id]
                costval = feature_cost_cuda(metric, data1, row_sums1, row_sqnorms1, row_means1, row_centered_sqnorms1, scale, tid_x, j + local_id, ncols)
            end
            local_acc += exp(muladd(costval, invreg, muval) - maxval)
        end
        local_acc = CUDA.CUDACore.reduce_warp(+, local_acc)
        if local_id == 0
            output[tid_x] = log(marginal[tid_x]) - (log(local_acc) + maxval)
        end
        tid_x += nwarps
    end
    return
end

function warp_logsumexp_cellsim_fused!(
    output::CuDeviceVector{T},
    data1::CuDeviceMatrix{T},
    data2::CuDeviceMatrix{T},
    row_sums1::CuDeviceVector{T},
    row_sqnorms1::CuDeviceVector{T},
    row_means1::CuDeviceVector{T},
    row_centered_sqnorms1::CuDeviceVector{T},
    row_sums2::CuDeviceVector{T},
    row_sqnorms2::CuDeviceVector{T},
    row_means2::CuDeviceVector{T},
    row_centered_sqnorms2::CuDeviceVector{T},
    scale::T,
    metric::Symbol,
    marginal::CuDeviceVector{T},
    θ::CuDeviceVector{T},
    reg::T,
    W∞::T,
) where T
    step = warpsize()
    nwarps = (gridDim().x * blockDim().x) ÷ step
    tid_x = (threadIdx().x + (blockIdx().x - 1) * blockDim().x - 1) ÷ step + 1
    N = size(data1, 1)
    M = size(data2, 1)
    ncols = size(data1, 2)
    N_outer = Int(ceil(N / nwarps))
    local_id = (threadIdx().x - 1) % step
    invreg = -one(T) / (reg * W∞)
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
                costval = feature_cost_cuda(metric, data1, row_sums1, row_sqnorms1, row_means1, row_centered_sqnorms1, scale, tid_x, j + local_id, ncols)
            end
            v = muladd(costval, invreg, muval)
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
                costval = feature_cost_cuda(metric, data1, row_sums1, row_sqnorms1, row_means1, row_centered_sqnorms1, scale, tid_x, j + local_id, ncols)
            end
            v = muladd(costval, invreg, muval)
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
            output[tid_x] = log(marginal[tid_x]) - (log(s) + m)
        end
        tid_x += nwarps
    end
    return
end

function sinkhorn_cellsim(
    data1::CuArray{T},
    data2::CuArray{T},
    scale::T,
    metric::Symbol,
    marginal1::CuArray{T},
    marginal2::CuArray{T},
    args::EOTArgs,
    frequency::Int=100;
    return_cuda=false,
    φ0::Union{CuArray{T},typeof(Nothing)}=Nothing,
    ψ0::Union{CuArray{T},typeof(Nothing)}=Nothing
) where T<:Real
    N = size(data1, 1)
    M = size(data2, 1)
    if φ0 == Nothing
        φ = CUDA.zeros(T, N)
        ψ = CUDA.zeros(T, M)
    else
        φ = φ0
        ψ = ψ0
    end
    residual_cache = CUDA.zeros(T, N)
    cost_cache = CUDA.zeros(T, N)
    row_sums1, row_sqnorms1, row_means1, row_centered_sqnorms1 = get_flat_metrics(data1)
    row_sums2, row_sqnorms2, row_means2, row_centered_sqnorms2 = get_flat_metrics(data2)
    threads = 256
    blocks = div(N, div(threads, 32, RoundDown), RoundUp)
    time_start = time_ns()
    @cuda threads = threads blocks = blocks max_logsumexp_cellsim!(residual_cache, data1, row_sums1, row_sqnorms1, row_means1, row_centered_sqnorms1, scale, metric)
    W∞ = maximum(residual_cache)
    η = args.eta_p
    num_iter = 0
    if args.verbose
        println("time(s),iter,infeas,ot_objective,dual")
    end
    for i in 1:args.itermax
        elapsed_time = (time_ns() - time_start) / 1e9
        if elapsed_time > args.tmax
            break
        end
        @cuda threads = threads blocks = blocks warp_logsumexp_cellsim_fused!(φ, data1, data2, row_sums1, row_sqnorms1, row_means1, row_centered_sqnorms1, row_sums2, row_sqnorms2, row_means2, row_centered_sqnorms2, scale, metric, marginal1, ψ, η, W∞)
        @cuda threads = threads blocks = div(M, div(threads, 32, RoundDown), RoundUp) warp_logsumexp_cellsim_fused!(ψ, data2, data1, row_sums2, row_sqnorms2, row_means2, row_centered_sqnorms2, row_sums1, row_sqnorms1, row_means1, row_centered_sqnorms1, scale, metric, marginal2, φ, η, W∞)
        CUDA.synchronize()
        if (i - 1) % frequency == 0
            @cuda threads = threads blocks = blocks residual_cellsim_opt!(residual_cache, cost_cache, data1, data2, row_sums1, row_sqnorms1, row_means1, row_centered_sqnorms1, row_sums2, row_sqnorms2, row_means2, row_centered_sqnorms2, scale, metric, marginal1, φ, ψ, η, W∞)
            CUDA.synchronize()

            residual_r = norm(residual_cache, 1)
            if args.verbose
                ot_objective = sum(cost_cache)
                objective = (dot(ψ, marginal2) + dot(φ, marginal1))
                @printf "%.6e,%d,%.14e,%.14e,%.14e,sinkhorn_cellsim\n" elapsed_time i residual_r ot_objective objective
            end
            if residual_r <= args.epsilon / 6
                break
            end
        end

        num_iter += 1
    end
    @cuda threads = threads blocks = blocks residual_cellsim_opt!(residual_cache, cost_cache, data1, data2, row_sums1, row_sqnorms1, row_means1, row_centered_sqnorms1, row_sums2, row_sqnorms2, row_means2, row_centered_sqnorms2, scale, metric, marginal1, φ, ψ, η, W∞)
    CUDA.synchronize()
    objective = sum(cost_cache)
    residual_val = sum(abs.(residual_cache))
    if return_cuda
        return φ, ψ, objective, residual_val, num_iter
    else
        return Array(φ), Array(ψ), objective, residual_val
    end
end
