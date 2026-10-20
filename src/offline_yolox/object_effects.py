"""Foreground surface effects without changing support masks or geometry."""
import cv2
import numpy as np


def apply_object_surface(image, mask, *, blur_sigma=0.0, dirt_strength=0.0,
                         dirt_color_bgr=(70, 100, 130), rng=None):
    """Add procedural earth-colored stains, then normalized masked Gaussian blur.

    Dirt is a color layer on the object surface, not an occluder. Its strength
    bounds local opacity. Blur uses mask-normalized weights to avoid drawing
    black/exterior asset pixels into the foreground. Outside pixels are retained.
    """
    if blur_sigma == 0 and dirt_strength == 0:
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
    if blur_sigma > 0:
        weights = mask.astype(np.float32)
        numerator = cv2.GaussianBlur(result.astype(np.float32) * weights[:, :, None],
                                     (0, 0), blur_sigma, borderType=cv2.BORDER_CONSTANT)
        denominator = cv2.GaussianBlur(weights, (0, 0), blur_sigma,
                                       borderType=cv2.BORDER_CONSTANT)
        blurred = numerator[mask] / np.maximum(denominator[mask, None], 1e-6)
        result[mask] = np.clip(np.rint(blurred), 0, 255).astype(np.uint8)
    return result
