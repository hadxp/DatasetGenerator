import numpy as np
from PIL import Image
from typing import List, Optional

def pil_to_numpy(
    pil_image: Image.Image,
    mode: str = "RGB",
    dtype: np.dtype = np.uint8,
    normalize: bool = False
) -> np.ndarray:
    """
    Convert PIL Image to NumPy array.

    Args:
        pil_image: PIL Image object
        mode: Target color mode ('RGB', 'BGR', 'GRAY', 'RGBA', etc.)
        dtype: Output data type (np.uint8, np.float32, etc.)
        normalize: If True and dtype is float, normalize to [0, 1]

    Returns:
        NumPy array
    """
    if not isinstance(pil_image, Image.Image):
        raise TypeError(f"Expected PIL Image, got {type(pil_image)}")

    # Store original mode for reference
    original_mode = pil_image.mode

    # Handle different conversions based on target mode
    if mode.upper() in ["BGR", "BGR8"]:
        # PIL uses RGB, so convert to RGB first then swap channels
        rgb_array = np.array(pil_image.convert("RGB"))
        # RGB -> BGR
        bgr_array = rgb_array[:, :, ::-1].copy()  # Use copy() for contiguous array
        result = bgr_array

    elif mode.upper() in ["GRAY", "GRAYSCALE", "L"]:
        if pil_image.mode == "RGBA" and original_mode == "RGBA":
            # For RGBA images, convert to RGB first to avoid alpha channel issues
            gray_array = np.array(pil_image.convert("RGB").convert("L"))
        else:
            gray_array = np.array(pil_image.convert("L"))
        result = gray_array

    elif mode.upper() == "RGB":
        result = np.array(pil_image.convert("RGB"))

    elif mode.upper() == "RGBA":
        if pil_image.mode == "RGBA":
            result = np.array(pil_image)
        else:
            # Add alpha channel
            rgb_array = np.array(pil_image.convert("RGB"))
            alpha_channel = np.full(rgb_array.shape[:2], 255, dtype=np.uint8)
            result = np.dstack((rgb_array, alpha_channel))

    else:
        # For other modes, let PIL handle conversion
        result = np.array(pil_image.convert(mode))

    # Handle dtype conversion
    if dtype != result.dtype:
        if dtype in [np.float32, np.float64]:
            if normalize:
                result = result.astype(dtype) / 255.0
            else:
                result = result.astype(dtype)
        else:
            result = result.astype(dtype)

    return result


def numpy_to_pil(
    numpy_array: np.ndarray,
    mode: Optional[str] = None
) -> Image.Image:
    """
    Convert NumPy array to PIL Image.

    Args:
        numpy_array: NumPy array with shape (H, W), (H, W, 1), (H, W, 3), or (H, W, 4)
        mode: PIL mode ('RGB', 'L', 'RGBA', 'BGR', etc.)
               If None, automatically detects based on array shape

    Returns:
        PIL Image object
    """
    if not isinstance(numpy_array, np.ndarray):
        raise TypeError(f"Expected NumPy array, got {type(numpy_array)}")

    # Handle empty arrays
    if numpy_array.size == 0:
        raise ValueError("Cannot convert empty array to PIL Image")

    # Remove singleton dimensions
    numpy_array = np.squeeze(numpy_array)

    # Determine shape and dimensions
    if numpy_array.ndim == 2:
        h, w = numpy_array.shape
        channels = 1
    elif numpy_array.ndim == 3:
        h, w, channels = numpy_array.shape
    else:
        raise ValueError(f"Unsupported array shape: {numpy_array.shape}")

    # Auto-detect mode if not specified
    if mode is None:
        if channels == 1:
            mode = "L"  # Grayscale
        elif channels == 3:
            # Check if it's BGR or RGB
            # Common assumption: if dtype is uint8 and array looks like BGR
            mode = "RGB"
        elif channels == 4:
            mode = "RGBA"
        else:
            raise ValueError(f"Cannot auto-detect mode for {channels} channels")

    # Handle BGR to RGB conversion
    if mode.upper() == "BGR":
        if channels != 3:
            raise ValueError("BGR mode requires 3 channels")
        # Convert BGR to RGB
        numpy_array = numpy_array[:, :, ::-1].copy()
        mode = "RGB"

    # Handle float arrays (normalized to [0, 1])
    if numpy_array.dtype in [np.float32, np.float64]:
        if numpy_array.max() <= 1.0 and numpy_array.min() >= 0:
            # Normalized float, convert to uint8
            numpy_array = (numpy_array * 255).astype(np.uint8)
        else:
            # Assume it's already in 0-255 range or similar
            numpy_array = numpy_array.astype(np.uint8)

    # Ensure correct dtype for PIL
    if numpy_array.dtype != np.uint8:
        numpy_array = numpy_array.astype(np.uint8)

    # Create PIL Image
    pil_image = Image.fromarray(numpy_array, mode=mode)

    return pil_image


def batch_pil_to_numpy(
    pil_images: List[Image.Image],
    mode: str = "RGB",
    dtype: np.dtype = np.uint8,
    normalize: bool = False
) -> List[np.ndarray]:
    """
    Convert a list of PIL Images to NumPy arrays.

    Args:
        pil_images: List of PIL Image objects
        mode: Target color mode
        dtype: Output data type
        normalize: Normalize float arrays to [0, 1]

    Returns:
        List of NumPy arrays
    """
    return [
        pil_to_numpy(img, mode=mode, dtype=dtype, normalize=normalize)
        for img in pil_images
    ]


def batch_numpy_to_pil(
    numpy_arrays: List[np.ndarray],
    mode: Optional[str] = None
) -> List[Image.Image]:
    """
    Convert a list of NumPy arrays to PIL Images.

    Args:
        numpy_arrays: List of NumPy arrays
        mode: PIL mode (auto-detected if None)

    Returns:
        List of PIL Image objects
    """
    return [
        numpy_to_pil(arr, mode=mode)
        for arr in numpy_arrays
    ]

