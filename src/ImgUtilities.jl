using CUDA
using Images
using KernelDensity


function flatten_image(img::AbstractArray{T}) where T<:Real
    if ndims(img) == 3
        channels, w, h = size(img)
    elseif ndims(img) == 2
        w, h = size(img)
        channels = 1
    else
        throw("Unsupported image shape $(size(img))")
    end
    original_shape = size(img)
    img_rearranged = reshape(img, (channels, w * h))

    return img_rearranged, original_shape
end

@inline function rgb_distance(pix1r, pix1g, pix1b, pix2r, pix2g, pix2b)
    return (pix1r - pix2r)^2 + (pix1g - pix2g)^2 + (pix1b - pix2b)^2
end

function restore_image(img::AbstractArray{T}, original_shape::Tuple) where T<:Real
    channels, w, h = original_shape
    return reshape(img, (channels, w, h))
end

function load_rgb(filepath::String; cuda::Bool=false, resolution::Tuple=(), uniform_marginal=false, dtype::Type=Float64)
    img = dtype.(channelview(imresize(load(filepath), resolution, ratio=1.0)))

    img, original_dims = flatten_image(img)
    if !uniform_marginal
        nearest_log2 = Base.ceil(log2(dtype(size(img, 2))))
        nearest_pow2 = Int(2^nearest_log2)
        Ur = kde(img[1, :], npoints=nearest_pow2)
        normalize!(Ur.density)
        Ug = kde(img[2, :], npoints=nearest_pow2)
        normalize!(Ug.density)
        Ub = kde(img[3, :], npoints=nearest_pow2)
        normalize!(Ub.density)

        marginal = dtype.(normalize(pdf(Ur, img[1, :]) + pdf(Ug, img[2, :]) + pdf(Ub, img[3, :]), 1))
    else
        marginal = ones(dtype, size(img, 2)) / size(img, 2)
    end
    if cuda
        return CuArray(img), original_dims, CuArray(marginal)
    end
    return img, original_dims, marginal
end
function apply_colormap(img::AbstractMatrix{T}, colormap::AbstractVector{Int}) where T<:Real
    imgcopy = img[:, colormap]
end
function save_image(path::String, flattened_image::AbstractArray{T}, original_shape::Tuple) where T<:Real
    img = colorview(RGB, restore_image(flattened_image, original_shape))
    save(path, img)
end

