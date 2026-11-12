"""Foreground surface effects without changing support masks or geometry."""
import cv2
import numpy as np


def apply_object_surface(image, mask, *, blur_sigma=0.0, dirt_strength=0.0, resolution_scale=1.0,
                         dirt_color_bgr=(70, 100, 130), rng=None):
    """Add stains, reduce effective resolution, then apply masked Gaussian blur.

    Dirt is a color layer on the object surface, not an occluder. Its strength
    bounds local opacity. Blur uses mask-normalized weights to avoid drawing
    black/exterior asset pixels into the foreground. Resolution reduction also
    resamples premultiplied colors and mask weights together. Final dimensions,
    binary support and outside pixels are retained.
    """
    if (isinstance(resolution_scale, bool) or not isinstance(resolution_scale, (int, float))
            or not np.isfinite(resolution_scale) or not 0 < resolution_scale <= 1):
        raise ValueError("resolution_scale must be finite and 0 < scale <= 1")
    if blur_sigma == 0 and dirt_strength == 0 and resolution_scale == 1:
        return image
    result = image.copy()
    if dirt_strength > 0:
        if rng is None:
            raise ValueError("Surface dirt requires an explicit seeded RNG")
        height, width = mask.shape
        coarse = cv2.resize(rng.random((6, 6), dtype=np.float32), (width, height),
                            interpolation=cv2.INTER_LINEAR)
        grain = cv2.resize(rng.random((24, 24), dtype=np.float32), (width, height),
                           interpolation=cv2.INTER_LINEAR)
        texture = np.clip((0.8 * coarse + 0.2 * grain - 0.3) / 0.5, 0, 1)
        alpha = (dirt_strength * texture[mask])[:, None]
        color = np.asarray(dirt_color_bgr, dtype=np.float32)
        stained = result[mask].astype(np.float32) * (1 - alpha) + color * alpha
        result[mask] = np.clip(np.rint(stained), 0, 255).astype(np.uint8)
    if resolution_scale < 1:
        height, width = mask.shape
        small_size = (max(1, int(round(width * resolution_scale))),
                      max(1, int(round(height * resolution_scale))))
        weights = mask.astype(np.float32)
        small_color = cv2.resize(result.astype(np.float32) * weights[:, :, None],
                                 small_size, interpolation=cv2.INTER_AREA)
        small_weights = cv2.resize(weights, small_size, interpolation=cv2.INTER_AREA)
        restored_color = cv2.resize(small_color, (width, height), interpolation=cv2.INTER_LINEAR)
        restored_weights = cv2.resize(small_weights, (width, height), interpolation=cv2.INTER_LINEAR)
        valid = mask & (restored_weights > 1e-8)
        reduced = restored_color[valid] / restored_weights[valid, None]
        result[valid] = np.clip(np.rint(reduced), 0, 255).astype(np.uint8)
    if blur_sigma > 0:
        weights = mask.astype(np.float32)
        numerator = cv2.GaussianBlur(result.astype(np.float32) * weights[:, :, None],
                                     (0, 0), blur_sigma, borderType=cv2.BORDER_CONSTANT)
        denominator = cv2.GaussianBlur(weights, (0, 0), blur_sigma,
                                       borderType=cv2.BORDER_CONSTANT)
        blurred = numerator[mask] / np.maximum(denominator[mask, None], 1e-6)
        result[mask] = np.clip(np.rint(blurred), 0, 255).astype(np.uint8)
    return result
