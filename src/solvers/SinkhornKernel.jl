using CUDA
using IterTools
using LinearAlgebra
using Test
using BenchmarkTools



function residual_opt!(output::CuDeviceVector{T}, cost_output::CuDeviceVector{T}, img1::CuDeviceMatrix{T},
    img2::CuDeviceMatrix{T}, marginal::CuDeviceVector{T}, φ::CuDeviceVector{T}, ψ::CuDeviceVector{T}, reg::T, W∞::T, p::T) where {T}
    step = warpsize()
    nwarps = (gridDim().x * blockDim().x) ÷ step
    tid_x = (threadIdx().x + (blockIdx().x - 1) * blockDim().x - 1) ÷ step + 1
    N = size(img1, 2)
    N_outer = Int(ceil(N / nwarps))
    local_id = (threadIdx().x - 1) % step
    invreg = -one(T) / (reg * W∞)
    Ntiles = (N) ÷ step
    for _ in 1:N_outer
        local_acc = T(0.0)
        cost_acc = T(0.0)
        @inbounds begin
            pix1r = img1[1, tid_x]
            pix1g = img1[2, tid_x]
            pix1b = img1[3, tid_x]
            φi = φ[tid_x]
        end
        for tile in 0:(Ntiles-1)
            j = tile * warpsize() + 1
            @inbounds begin
                pix2r = img2[1, j+local_id]
                pix2g = img2[2, j+local_id]
                pix2b = img2[3, j+local_id]
                muval = ψ[j+local_id]
                dr = (pix1r - pix2r)
                dg = (pix1g - pix2g)
                db = (pix1b - pix2b)
                if p == 1
                    l2dist = abs(dr) + abs(dg) + abs(db)
                else
                    l2dist = muladd(dr, dr, muladd(dg, dg, muladd(db, db, 0)))
                end
                value = muladd(l2dist, invreg, φi) + muval
            end

            local_acc += exp(value)
            cost_acc += exp(value) * l2dist
        end
        if (Ntiles) * warpsize() + local_id < N
            j = (Ntiles) * warpsize() + 1
            @inbounds begin
                pix2r = img2[1, j+local_id]
                pix2g = img2[2, j+local_id]
                pix2b = img2[3, j+local_id]
                muval = ψ[j+local_id]
                dr = (pix1r - pix2r)
                dg = (pix1g - pix2g)
                db = (pix1b - pix2b)
                if p == 1
                    l2dist = abs(dr) + abs(dg) + abs(db)
                else
                    l2dist = muladd(dr, dr, muladd(dg, dg, muladd(db, db, 0)))
                end
                value = muladd(l2dist, invreg, φi) + muval
            end

            local_acc += exp(value)
            cost_acc += exp(value) * l2dist
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


function residual!(output::CuDeviceVector{T}, img1::CuDeviceMatrix{T},
    img2::CuDeviceMatrix{T}, φ::CuDeviceVector{T}, ψ::CuDeviceVector{T}, reg::T, p::T) where {T}
    step = warpsize()
    tid_x = (threadIdx().x + (blockIdx().x - 1) * blockDim().x - 1) ÷ step + 1
    N = size(img1, 2)
    if tid_x > N
        return
    end
    local_id = (threadIdx().x - 1) % step
    pix1r = img1[1, tid_x]
    pix1g = img1[2, tid_x]
    pix1b = img1[3, tid_x]
    φi = φ[tid_x]
    local_acc = T(0.0)
    for i in 1:step:N
        if i + local_id > N
            break
        end
        pix2r = img2[1, i+local_id]
        pix2g = img2[2, i+local_id]
        pix2b = img2[3, i+local_id]
        if p == 1
            l2dist = abs(dr) + abs(dg) + abs(db)
        else
            l2dist = muladd(dr, dr, muladd(dg, dg, muladd(db, db, 0)))
        end
        value = -(l2dist) / reg + φi + ψ[i+local_id]
        local_acc += exp(value)
    end
    local_acc = CUDA.CUDACore.reduce_warp(+, local_acc)
    if local_id == 0
        output[tid_x] = 1 / N - local_acc
    end
    return
end


@inline function eot_exponent(pix1r, pix1g, pix1b, pix2r, pix2g, pix2b, φi, ψj, η)
    return -(rgb_distance(pix1r, pix1g, pix1b, pix2r, pix2g, pix2b)) / reg + ψj + φi + η
end
function naive_findmaxindex_ct!(output_img::CuDeviceMatrix{T}, img1::CuDeviceMatrix{T},
    img2::CuDeviceMatrix{T}, φ::CuDeviceVector{T}, ψ::CuDeviceVector{T}, reg::T, W∞::T, p::T) where {T}
    tid_x = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    N = size(img1, 2)
    if tid_x > N
        return
    end
    pix1r = img1[1, tid_x]
    pix1g = img1[2, tid_x]
    pix1b = img1[3, tid_x]
    φi = φ[tid_x]
    avgr = T(0.0)
    avgg = T(0.0)
    avgb = T(0.0)
    probsum = T(0.0)
    for i in 1:N
        pix2r = img2[1, i]
        pix2g = img2[2, i]
        pix2b = img2[3, i]
        dr = pix1r - pix2r
        dg = pix1g - pix2g
        db = pix1b - pix2b
        if p == 1
            l2dist = abs(dr) + abs(dg) + abs(db)
        else
            l2dist = muladd(dr, dr, muladd(dg, dg, muladd(db, db, 0)))
        end
        prob = exp(-(l2dist) / (reg * W∞) + φi + ψ[i])
        probsum += prob
        avgr += img2[1, i] * prob
        avgg += img2[2, i] * prob
        avgb += img2[3, i] * prob
    end
    output_img[1, tid_x] = avgr / probsum
    output_img[2, tid_x] = avgg / probsum
    output_img[3, tid_x] = avgb / probsum

    return
end

function warp_logsumexp_ct!(output::CuDeviceVector{T}, img1::CuDeviceMatrix{T},
    img2::CuDeviceMatrix{T}, ψ::CuDeviceVector{T}, reg::T, p::T) where {T}
    step = warpsize()
    nwarps = (gridDim().x * blockDim().x) ÷ step
    tid_x = (threadIdx().x + (blockIdx().x - 1) * blockDim().x - 1) ÷ step + 1
    N = size(img1, 2)
    N_outer = Int(ceil(N / nwarps))
    local_id = (threadIdx().x - 1) % step
    for i in 1:N_outer
        if tid_x > N
            return
        end
        pix1r = img1[1, tid_x]
        pix1g = img1[2, tid_x]
        pix1b = img1[3, tid_x]
        maxval = -Inf
        for i in 1:step:N
            if i + local_id > N
                break
            end
            pix2r = img2[1, i+local_id]
            pix2g = img2[2, i+local_id]
            pix2b = img2[3, i+local_id]
            dr = pix1r - pix2r
            dg = pix1g - pix2g
            db = pix1b - pix2b
            if p == 1
                l2dist = abs(dr) + abs(dg) + abs(db)
            else
                l2dist = muladd(dr, dr, muladd(dg, dg, muladd(db, db, 0)))
            end
            value = -(l2dist) / reg + ψ[i+local_id]
            maxval = max(value, maxval)
        end
        maxval = CUDA.CUDACore.reduce_warp(max, maxval)
        maxval = shfl_sync(CUDA.FULL_MASK, maxval, 1)
        local_acc = T(0.0)
        for i in 1:step:N
            if i + local_id > N
                break
            end
            pix2r = img2[1, i+local_id]
            pix2g = img2[2, i+local_id]
            pix2b = img2[3, i+local_id]
            dr = pix1r - pix2r
            dg = pix1g - pix2g
            db = pix1b - pix2b

            if p == 1
                l2dist = abs(dr) + abs(dg) + abs(db)
            else
                l2dist = muladd(dr, dr, muladd(dg, dg, muladd(db, db, 0)))
            end
            value = -(l2dist) / reg + ψ[i+local_id]
            local_acc += exp(value - maxval)
        end
        local_acc2 = CUDA.CUDACore.reduce_warp(+, local_acc)
        sync_warp()
        if local_id == 0
            output[tid_x] = -log(N) - (log(local_acc2) + maxval)
        end
        tid_x += nwarps
    end
    return
end

function warp_logsumexp_ct_opt!(output::CuDeviceVector{T}, img1::CuDeviceMatrix{T},
    img2::CuDeviceMatrix{T}, marginal::CuDeviceVector{T}, ψ::CuDeviceVector{T}, reg::T, W∞::T, p::T) where {T}
    step = warpsize()
    nwarps = (gridDim().x * blockDim().x) ÷ step
    tid_x = (threadIdx().x + (blockIdx().x - 1) * blockDim().x - 1) ÷ step + 1
    N = size(img1, 2)
    N_outer = Int(ceil(N / nwarps))
    local_id = (threadIdx().x - 1) % step
    invreg = -one(T) / (reg * W∞)
    Ntiles = (N) ÷ step
    for _ in 1:N_outer
        if tid_x > N
            return
        end
        pix1r = img1[1, tid_x]
        pix1g = img1[2, tid_x]
        pix1b = img1[3, tid_x]
        maxval = -Inf

        for tile in 0:(Ntiles-1)
            j = tile * warpsize() + 1
            @inbounds begin
                pix2r = img2[1, j+local_id]
                pix2g = img2[2, j+local_id]
                pix2b = img2[3, j+local_id]
                muval = ψ[j+local_id]
                dr = (pix1r - pix2r)
                dg = (pix1g - pix2g)
                db = (pix1b - pix2b)
                if p == 1
                    l2dist = abs(dr) + abs(dg) + abs(db)
                else
                    l2dist = muladd(dr, dr, muladd(dg, dg, muladd(db, db, 0)))
                end
                value = muladd(l2dist, invreg, muval)
            end
            maxval = max(value, maxval)
        end
        if (Ntiles) * warpsize() + local_id < N
            j = (Ntiles) * warpsize() + 1
            @inbounds begin
                pix2r = img2[1, j+local_id]
                pix2g = img2[2, j+local_id]
                pix2b = img2[3, j+local_id]
                muval = ψ[j+local_id]
                dr = (pix1r - pix2r)
                dg = (pix1g - pix2g)
                db = (pix1b - pix2b)
                if p == 1
                    l2dist = abs(dr) + abs(dg) + abs(db)
                else
                    l2dist = muladd(dr, dr, muladd(dg, dg, muladd(db, db, 0)))
                end
                value = muladd(l2dist, invreg, muval)
            end
            maxval = max(value, maxval)
        end
        maxval = CUDA.CUDACore.reduce_warp(max, maxval)
        maxval = CUDA.shfl_sync(CUDA.FULL_MASK, maxval, 1)

        local_acc = T(0.0)
        for tile in 0:(Ntiles-1)
            j = tile * warpsize() + 1
            @inbounds begin
                pix2r = img2[1, j+local_id]
                pix2g = img2[2, j+local_id]
                pix2b = img2[3, j+local_id]
                muval = ψ[j+local_id]
                dr = (pix1r - pix2r)
                dg = (pix1g - pix2g)
                db = (pix1b - pix2b)
                if p == 1
                    l2dist = abs(dr) + abs(dg) + abs(db)
                else
                    l2dist = muladd(dr, dr, muladd(dg, dg, muladd(db, db, 0)))
                end
                value = muladd(l2dist, invreg, muval)
            end

            local_acc += exp(value - maxval)
        end
        if (Ntiles) * warpsize() + local_id < N
            j = (Ntiles) * warpsize() + 1
            @inbounds begin
                pix2r = img2[1, j+local_id]
                pix2g = img2[2, j+local_id]
                pix2b = img2[3, j+local_id]
                muval = ψ[j+local_id]
                dr = (pix1r - pix2r)
                dg = (pix1g - pix2g)
                db = (pix1b - pix2b)
                if p == 1
                    l2dist = abs(dr) + abs(dg) + abs(db)
                else
                    l2dist = muladd(dr, dr, muladd(dg, dg, muladd(db, db, 0)))
                end
                value = muladd(l2dist, invreg, muval)
            end

            local_acc += exp(value - maxval)
        end
        local_acc = CUDA.CUDACore.reduce_warp(+, local_acc)
        if local_id == 0
            output[tid_x] = log(marginal[tid_x]) - (log(local_acc) + maxval)
        end
        tid_x += nwarps
    end
    return
end


function warp_logsumexp_ct_fused!(output::CuDeviceVector{T}, img1::CuDeviceMatrix{T},
    img2::CuDeviceMatrix{T}, marginal::CuDeviceVector{T}, θ::CuDeviceVector{T}, reg::T, W∞::T, p::T) where T
    step = warpsize()
    nwarps = (gridDim().x * blockDim().x) ÷ step
    tid_x = (threadIdx().x + (blockIdx().x - 1) * blockDim().x - 1) ÷ step + 1
    N = size(img1, 2)
    M = size(img2, 2)
    N_outer = Int(ceil(N / nwarps))
    local_id = (threadIdx().x - 1) % step
    invreg = -one(T) / (reg * W∞)
    Ntiles = (M) ÷ step

    for _ in 1:N_outer
        if tid_x > N
            return
        end
        @inbounds begin
            pix1r = img1[1, tid_x]
            pix1g = img1[2, tid_x]
            pix1b = img1[3, tid_x]
        end
        m_local = T(-Inf)
        s_local = T(0)

        for tile in 0:(Ntiles-1)
            j = tile * warpsize() + 1
            @inbounds begin
                pix2r = img2[1, j+local_id]
                pix2g = img2[2, j+local_id]
                pix2b = img2[3, j+local_id]
                muval = θ[j+local_id]
                dr = (pix1r - pix2r)
                dg = (pix1g - pix2g)
                db = (pix1b - pix2b)
                if p == 1
                    l2dist = abs(dr) + abs(dg) + abs(db)
                elseif p == 2
                    l2dist = muladd(dr, dr, muladd(dg, dg, muladd(db, db, 0)))
                elseif p == Inf
                    l2dist = max(abs(dr), max(abs(dg), abs(db)))
                else
                    l2dist = abs(dr)^p + abs(dg)^p + abs(db)^p
                    l2dist = abs(dr)^p + abs(dg)^p + abs(db)^p
                end
            end
            v = muladd(l2dist, invreg, muval)
            if v <= m_local
                s_local += exp(v - m_local)
            else
                s_local = s_local * exp(m_local - v) + one(T)
                m_local = v
            end
        end
        if (Ntiles) * warpsize() + local_id < M
            j = (Ntiles) * warpsize() + 1
            @inbounds begin
                pix2r = img2[1, j+local_id]
                pix2g = img2[2, j+local_id]
                pix2b = img2[3, j+local_id]
                muval = θ[j+local_id]
                dr = (pix1r - pix2r)
                dg = (pix1g - pix2g)
                db = (pix1b - pix2b)
                if p == 1
                    l2dist = abs(dr) + abs(dg) + abs(db)
                elseif p == 2
                    l2dist = muladd(dr, dr, muladd(dg, dg, muladd(db, db, 0)))
                elseif p == Inf
                    l2dist = max(abs(dr), max(abs(dg), abs(db)))
                else
                    l2dist = abs(dr)^p + abs(dg)^p + abs(db)^p
                end
            end
            v = muladd(l2dist, invreg, muval)
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
            output[tid_x] = log(marginal[tid_x]) - (log(s) + m)
        end
        tid_x += nwarps
    end
    return
end

@inline function reduce_block(op, val::T, neutral, shared) where T
    threads = blockDim().x
    thread = threadIdx().x

    # shared mem for a complete reduction
    @inbounds shared[thread] = val

    # perform a reduction
    d = 1
    while d < threads
        sync_threads()
        index = 2 * d * (thread - 1) + 1
        @inbounds if index <= threads
            other_val = if index + d <= threads
                shared[index+d]
            else
                neutral
            end
            shared[index] = op(shared[index], other_val)
        end
        d *= 2
    end

    # load the final value on the first thread
    if thread == 1
        val = @inbounds shared[thread]
    end

    return val
end

function sinkhorn_color_transfer(
    img1::CuArray{T},
    img2::CuArray{T},
    marginal1::CuArray{T},
    marginal2::CuArray{T},
    args::EOTArgs,
    frequency::Int=100,
    p::Union{Int,Float64}=2;
    return_cuda::Bool=false,
    return_assignments::Bool=true,
    φ0::Union{CuArray{T},Nothing}=nothing,
    ψ0::Union{CuArray{T},Nothing}=nothing,
) where T<:Real
    p_kernel = T(p)
    N = size(img1, 2)
    φ = isnothing(φ0) ? CUDA.zeros(T, N) : copy(φ0)
    ψ = isnothing(ψ0) ? CUDA.zeros(T, N) : copy(ψ0)
    residual_cache = CUDA.zeros(T, N)
    cost_cache = CUDA.zeros(T, N)
    threads = 256
    blocks = div(N, div(threads, 32, RoundDown), RoundUp)
    time_start = time_ns()
    η = args.eta_p
    @cuda threads = threads blocks = blocks max_logsumexp_spp_ct!(residual_cache, img1, img2, p)
    CUDA.synchronize()
    W∞ = maximum(residual_cache)
    # if p == Inf
    #     img1 ./= W∞
    #     img2 ./= W∞
    # else
    #     W∞ = 1.0
    # end
    if args.verbose
        println("time(s),iter,infeas,ot_objective,dual")
    end
    num_iter = 0
    for i in 1:args.itermax
        elapsed_time = (time_ns() - time_start) / 1e9
        if elapsed_time > args.tmax
            break
        end
        if p == Inf
            @cuda threads = threads blocks = blocks warp_logsumexp_ct_opt!(φ, img1, img2, marginal1, ψ, η, W∞, p_kernel)
            @cuda threads = threads blocks = blocks warp_logsumexp_ct_opt!(ψ, img2, img1, marginal2, φ, η, W∞, p_kernel)
        else
            @cuda threads = threads blocks = blocks warp_logsumexp_ct_fused!(φ, img1, img2, marginal1, ψ, η, W∞, p_kernel)
            @cuda threads = threads blocks = blocks warp_logsumexp_ct_fused!(ψ, img2, img1, marginal2, φ, η, W∞, p_kernel)
        end
        CUDA.synchronize()
        if (i - 1) % frequency == 0
            @cuda threads = threads blocks = blocks residual_opt!(residual_cache, cost_cache, img1, img2, marginal1, φ, ψ, η, W∞, p_kernel)
            CUDA.synchronize()
            residual_r = norm(residual_cache, 1)
            if args.verbose
                ot_objective = sum(cost_cache)
                objective = (dot(ψ, marginal2) + dot(φ, marginal1))
                @printf "%.6e,%d,%.14e,%.14e,%.14e,sinkhorn_kernel\n" elapsed_time i residual_r ot_objective objective
            end
            if residual_r <= args.epsilon / 6
                break
            end
        end
        num_iter += 1
    end
    @cuda threads = threads blocks = blocks residual_opt!(residual_cache, cost_cache, img1, img2, marginal1, φ, ψ, η, W∞, p_kernel)
    CUDA.synchronize()
    residual_val = norm(residual_cache, 1)
    objective = sum(cost_cache)
    if !return_assignments
        if return_cuda
            return φ, ψ, objective, residual_val, num_iter
        end
        return Array(φ), Array(ψ), objective, residual_val
    end
    output_img1 = CUDA.zeros(T, 3, N)
    output_img2 = CUDA.zeros(T, 3, N)
    naive_blocks = div(N, threads, RoundUp)
    @cuda threads = threads blocks = naive_blocks naive_findmaxindex_ct!(output_img1, img1, img2, φ, ψ, η, W∞, p_kernel)
    @cuda threads = threads blocks = naive_blocks naive_findmaxindex_ct!(output_img2, img2, img1, ψ, φ, η, W∞, p_kernel)

    return Array(φ), Array(ψ), Array(output_img1), Array(output_img2)

end
function test_sinkhorn()
    N = 2^17
    img1 = CUDA.rand(Float64, 3, N)
    img2 = CUDA.rand(Float64, 3, N)
    η = 1e-4
    maxiter = 4
    sinkhorn_color_transfer(img1, img2, η, maxiter, 100)
end

function sinkhorn_color_transfer(f1::String, f2::String, out_f1::String, out_f2::String, resolution::Tuple{Int,Int}, args::EOTArgs, frequency::Int, p::Union{Float64,Int}; dtype::Type=Float64)
    img1, dims1, marginal1 = load_rgb(f1; cuda=true, resolution=resolution, dtype=dtype)
    img2, dims2, marginal2 = load_rgb(f2; cuda=true, resolution=resolution, dtype=dtype)
    _, _, img1_new, img2_new = sinkhorn_color_transfer(img1, img2, marginal1, marginal2, args, frequency, p)
    save_image(out_f1, img1_new, dims1)
    save_image(out_f2, img2_new, dims2)
end



function sinkhorn_euclidean(marginal1::CuArray{T}, marginal2::CuArray{T}, location1::CuArray{T}, location2::CuArray{T}, out1::String, out2::String, potentials::String, args::EOTArgs, frequency::Int, p::R) where {T,R}
    φ, ψ, assignments1, assignments2 = sinkhorn_color_transfer(location1, location2, marginal1, marginal2, args, frequency, p)
    if potentials != ""
        open(potentials * ".row", "w") do outfile
            for φi in φ
                println(outfile, φi)
            end
        end
        open(potentials * ".col", "w") do outfile
            # println(outfile, "")
            for ψj in ψ
                println(outfile, ψj)
            end
        end
    end
    if out1 != ""
        open(out1, "w") do outfile
            for index in eachindex(assignments1)
                println(outfile, "$(assignments1),$(index)")
            end
        end
    end
    if out2 != ""
        open(out2, "w") do outfile
            for index in eachindex(assignments2)
                println(outfile, "$(assignments2),$(index)")
            end
        end
    end
end
