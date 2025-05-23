from ..utils.logging import setup_log

logger = setup_log(__name__)

import numpy as np
from numba import njit
from astropy.constants import c
from scipy.special import erfc, erf


@njit
def rotational_broadening_kernel(delta_lambda_l, delta_lambda_array):
    """
    Returns a normalized rotational broadening kernel defined over delta_lambda_array.

    :delta_lambda_l: Maximum wavelength shift due to rotation (in same units as delta_lambda_array).
    :delta_lambda_array: Array of wavelength shifts (in same units as delta_lambda_l).
    The kernel is defined as:
    G = 2 / (pi * delta_lambda_l) * sqrt(1 - (delta_lambda_array / delta_lambda_l)^2)
    The kernel is normalized to unity integral.
    The kernel is symmetric around 0.
    The kernel is defined for delta_lambda_array values between -delta_lambda_l and +delta_lambda_l.
    """
    G = np.zeros_like(delta_lambda_array)
    G = (
        2
        / (np.pi * delta_lambda_l)
        * np.sqrt(1 - (delta_lambda_array / delta_lambda_l) ** 2)
    )
    G /= np.sum(G)
    return G


def rotational_broaden_onekernel(wavelength, flux, vsini, edge_handling="firstlast"):
    """
    This function applies rotational broadening to a spectrum using a single kernel.
    It assumes that the kernel is the same over the whole spectrum.
    However, the kernel is wavelength dependent .
    Here, the kernel is calculated using the maximum wavelength shift due to rotation.
    This method is very fast but if the wavelength range is large, it may not be accurate.
    """
    if not np.allclose(np.diff(wavelength), wavelength[1] - wavelength[0], atol=1e-6):
        raise ValueError("Wavelength array must be evenly spaced.")
    if vsini <= 0.0:
        raise ValueError("vsini must be positive.")
    #    if not (0.0 <= epsilon <= 1.0):
    #        raise ValueError("epsilon must be between 0 and 1.")

    bin_width = wavelength[1] - wavelength[0]
    vc = (vsini / c).decompose().value

    # wavelength shifts on limbs (largest velocity shift)
    # take maximum to always integrate over necessary length
    delta_lambda_L_max = np.mean(wavelength * vc)

    # symmetric array centered around 0, spaced by bin_width,
    # ranging approximately from -delta_lambda_L_max to +delta_lambda_L_max,
    N = int(np.floor(delta_lambda_L_max / bin_width))
    delta_lambda = np.arange(-N, N + 1) * bin_width
    rot_kernel = rotational_broadening_kernel(delta_lambda_L_max, delta_lambda)

    convolved_flux = np.convolve(flux, rot_kernel, mode="same")
    convolved_one = np.convolve(np.ones_like(flux), rot_kernel, mode="same")

    # Normalize to preserve flux
    sp_broadenend = convolved_flux / convolved_one

    # When applying convolution with wide kernel (e.g. here rotational broadening kernel),
    # values near the edges of  spectrum get corrupted
    # by the kernel. To avoid this, we need to extend the spectrum
    if edge_handling == "firstlast":
        # extend the spectrum by padding it with the first and last values
        # makes sense for non-normalized spectra which will be extended by continuum values on edges
        pad = int(np.floor(vc * np.max(wavelength) / bin_width)) + 1
        flux_ext = np.concatenate([np.full(pad, flux[0]), flux, np.full(pad, flux[-1])])
        out_idx = np.arange(len(flux)) + pad
    elif edge_handling == "normalized":
        # extend the spectrum by padding it with 1 on both sides
        # makes sense for normalized spectra which will be extended by continuum (1.0) values on edges
        pad = int(np.floor(vc * np.max(wavelength) / bin_width)) + 1
        flux_ext = np.concatenate([np.full(pad, 1.0), flux, np.full(pad, 1.0)])
        out_idx = np.arange(len(flux)) + pad
    else:
        raise ValueError(f"Unsupported edge_handling: {edge_handling}")
    sp_broadenend = np.convolve(flux_ext, rot_kernel, mode="same")

    return sp_broadenend[out_idx]


def rotational_broaden_chunks(wavelength, flux, vsini, epsilon, edge_handling):
    """
    np.convolve assumes the same kernel over the whole spectrum
    However, the kernel depends on the wavelength (delta_lambda_L=lambda/c* vsini)
    If wavelength range is short, we can use a single kernel
    If not, we need to split the spectrum into chunks
    and convolve each chunk with its own kernel

    # The number of chunks may need to be adjusted
    # depending on the vsini and the wavelength range
    """
    if not np.allclose(np.diff(wavelength), wavelength[1] - wavelength[0], atol=1e-6):
        raise ValueError("Wavelength array must be evenly spaced.")
    if vsini <= 0.0:
        raise ValueError("vsini must be positive.")
    #    if not (0.0 <= epsilon <= 1.0):
    #        raise ValueError("epsilon must be between 0 and 1.")

    bin_width = wavelength[1] - wavelength[0]
    vc = (vsini / c).decompose().value

    # np.convolve assumes the same kernel over the whole spectrum
    # However, the kernel depends on the wavelength (delta_lambda_L=lambda/c* vsini)
    # Solution: divide the spectrum into n_chunks
    # and convolve each chunk with its own kernel

    # Calculate kernel widths at chunk centers
    delta_lambda_L_array = wavelength * vc

    # total relative variation over full wavelength range
    rel_var_total = (
        np.max(delta_lambda_L_array) - np.min(delta_lambda_L_array)
    ) / np.mean(delta_lambda_L_array)

    # Estimate chunks assuming variation evenly split across chunks
    n_chunks = int(np.ceil(rel_var_total / epsilon))
    logger.debug(f"Divide spectrum into {n_chunks} chunks")

    # Define coarse wavelength grid for chunks
    wave_chunks = np.linspace(wavelength[0], wavelength[-1], n_chunks)

    # Global extension of wavelength and flux to avoid edge effects
    pad = int(np.ceil(vc * np.max(wavelength) / bin_width)) + 1
    if edge_handling == "firstlast":
        # extend the spectrum by padding it with the first and last values
        # makes sense for non-normalized spectra which will be extended by continuum values on edges
        flux_ext = np.concatenate([np.full(pad, flux[0]), flux, np.full(pad, flux[-1])])
        wave_ext = np.concatenate(
            [
                wavelength[0] + bin_width * (np.arange(-pad, 0)),
                wavelength,
                wavelength[-1] + bin_width * (np.arange(1, pad + 1)),
            ]
        )
    elif edge_handling == "normalized":
        # extend the spectrum by padding it with 1 on both sides
        # makes sense for normalized spectra which will be extended by continuum (1.0) values on edges
        flux_ext = np.concatenate([np.full(pad, 1.0), flux, np.full(pad, 1.0)])
        wave_ext = np.concatenate(
            [
                wavelength[0] + bin_width * (np.arange(-pad, 0)),
                wavelength,
                wavelength[-1] + bin_width * (np.arange(1, pad + 1)),
            ]
        )
    else:
        raise ValueError(f"Unsupported edge_handling: {edge_handling}")

    # Prepare output array - use normalized flux (1)
    broadened_flux = []

    # iterate over chunks
    for i in range(len(wave_chunks) - 1):
        L0 = wave_chunks[i]
        L1 = wave_chunks[i + 1]
        lambda_mid = 0.5 * (L0 + L1)
        # maximum wavelength shift at midpoint of chunk
        delta_lambda_L = lambda_mid * vc

        chunk_size = L1 - L0
        ext_width = 2.5 * delta_lambda_L  # extend beyond chunk by ~kernel width

        chunk_mask = (wave_ext >= L0) & (wave_ext < L1)
        chunk_with_ext_mask = (wave_ext >= L0 - ext_width) & (wave_ext < L1 + ext_width)

        flux_chunk_ext = flux_ext[chunk_with_ext_mask]

        # Define convolution kernel
        N = int(np.floor(delta_lambda_L / bin_width))
        delta_lambda = np.arange(-N, N + 1) * bin_width
        kernel = rotational_broadening_kernel(delta_lambda_L, delta_lambda)

        # Convolve
        convolved_flux = np.convolve(flux_chunk_ext, kernel, mode="same")
        convolved_one = np.convolve(np.ones_like(flux_chunk_ext), kernel, mode="same")
        norm_flux = convolved_flux / convolved_one

        # Cut back to inner chunk region
        idx_in_chunk = np.where(chunk_mask & chunk_with_ext_mask)[0]
        idx_start = idx_in_chunk[0] - np.where(chunk_with_ext_mask)[0][0]
        idx_end = idx_in_chunk[-1] - np.where(chunk_with_ext_mask)[0][0] + 1

        broadened_flux.extend(norm_flux[idx_start:idx_end])

    # Make sure the output is the same length as the input
    target_len = len(wavelength)
    if len(broadened_flux) < target_len:
        logger.warning(
            f"WARNING: broadened_flux ({len(broadened_flux)}) is shorter than target length ({target_len}). Padding with last value."
        )
        # Pad the last value to match length
        pad_size = target_len - len(broadened_flux)
        broadened_flux = np.concatenate(
            [broadened_flux, np.full(pad_size, broadened_flux[-1])]
        )

    elif len(broadened_flux) > target_len:
        logger.warning(
            f"WARNING: broadened_flux ({len(broadened_flux)}) is longer than target length ({target_len}). Trim excess values."
        )
        # Trim excess values
        broadened_flux = broadened_flux[:target_len]

    return broadened_flux


@njit
def gaussian_kernel(x, fwhm):
    """
    Gaussian kernel over wavelength offset x (same units as fwhm).
    """
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    gauss = np.exp(-0.5 * (x / sigma) ** 2)
    return gauss / np.sum(gauss)


def gaussian_broaden(wavelength, flux, fwhm, edge_handling="firstlast"):
    """
    Applies Gaussian broadening to a spectrum using a Gaussian kernel.
    The kernel is defined over the wavelength range of the spectrum.
    The width of the kernel is defined by fwhm.
    """
    if not np.allclose(np.diff(wavelength), wavelength[1] - wavelength[0], atol=1e-6):
        raise ValueError("Wavelength array must be evenly spaced.")
    if fwhm <= 0.0:
        raise ValueError("fwhm must be positive.")

    bin_width = wavelength[1] - wavelength[0]

    # Define Gaussian kernel
    delta_lambda = np.arange(-5 * fwhm, 5 * fwhm + bin_width, bin_width)
    kernel = gaussian_kernel(delta_lambda, fwhm)

    # Convolve
    convolved_flux = np.convolve(flux, kernel, mode="same")
    convolved_one = np.convolve(np.ones_like(flux), kernel, mode="same")

    # Normalize to preserve flux
    sp_broadenend = convolved_flux / convolved_one

    # When applying convolution with wide kernel (e.g. here rotational broadening kernel),
    # values near the edges of  spectrum get corrupted
    # by the kernel. To avoid this, we need to extend the spectrum
    if edge_handling == "firstlast":
        # extend the spectrum by padding it with the first and last values
        # makes sense for non-normalized spectra which will be extended by continuum values on edges
        pad = int(np.floor(fwhm / bin_width)) + 1
        flux_ext = np.concatenate([np.full(pad, flux[0]), flux, np.full(pad, flux[-1])])
        out_idx = np.arange(len(flux)) + pad
    elif edge_handling == "normalized":
        # extend the spectrum by padding it with 1 on both sides
        # makes sense for normalized spectra which will be extended by continuum (1.0) values on edges
        pad = int(np.floor(fwhm / bin_width)) + 1
        flux_ext = np.concatenate([np.full(pad, 1.0), flux, np.full(pad, 1.0)])
        out_idx = np.arange(len(flux)) + pad
    else:
        raise ValueError(f"Unsupported edge_handling: {edge_handling}")
    sp_broadenend = np.convolve(flux_ext, kernel, mode="same")

    return sp_broadenend[out_idx]


def macroturbulence_kernel(delta_lambda_l, delta_lambda_array):
    """
    Returns a normalized macro-turbulence kernel with
    radial-tangential broadening.
    """
    x = delta_lambda_array / delta_lambda_l

    G = np.exp(-x * x) + np.sqrt(np.pi) * np.abs(x) * (erf(np.abs(x)) - 1.0)

    G_sum = np.sum(G)
    if G_sum > 0:
        G /= G_sum

    return G


def macroturbulence_broaden_onekernel(
    wavelength, flux, vmac, edge_handling="firstlast"
):
    """
    Apply macro-turbulence broadening with a fixed kernel across the whole spectrum.
    """
    if not np.allclose(np.diff(wavelength), wavelength[1] - wavelength[0], atol=1e-6):
        raise ValueError("Wavelength array must be evenly spaced.")
    if vmac <= 0.0:
        raise ValueError("vmac must be positive.")

    bin_width = wavelength[1] - wavelength[0]
    vc = (vmac / c).decompose().value
    delta_lambda_L_max = np.mean(wavelength * vc)

    # symmetric array centered around 0, spaced by bin_width,
    # ranging approximately from -delta_lambda_L_max to +delta_lambda_L_max,
    N = int(np.floor(delta_lambda_L_max / bin_width))
    delta_lambda = np.arange(-N, N + 1) * bin_width
    kernel = macroturbulence_kernel(delta_lambda_L_max, delta_lambda)

    # When applying convolution with wide kernel (e.g. here rotational broadening kernel),
    # values near the edges of  spectrum get corrupted
    # by the kernel. To avoid this, we need to extend the spectrum
    pad = N + 1
    if edge_handling == "firstlast":
        flux_ext = np.concatenate([np.full(pad, flux[0]), flux, np.full(pad, flux[-1])])
    elif edge_handling == "normalized":
        flux_ext = np.concatenate([np.full(pad, 1.0), flux, np.full(pad, 1.0)])
    else:
        raise ValueError(f"Unsupported edge_handling: {edge_handling}")

    sp_broadened = np.convolve(flux_ext, kernel, mode="same")[pad:-pad]
    return sp_broadened


def macroturbulence_broaden_chunks(
    wavelength, flux, vmac, epsilon, edge_handling="firstlast"
):
    """
    np.convolve assumes the same kernel over the whole spectrum
    However, the kernel depends on the wavelength (delta_lambda_L=lambda/c* vsini)
    If wavelength range is short, we can use a single kernel
    If not, we need to split the spectrum into chunks
    and convolve each chunk with its own kernel

    """
    if not np.allclose(np.diff(wavelength), wavelength[1] - wavelength[0], atol=1e-6):
        raise ValueError("Wavelength array must be evenly spaced.")
    if vmac <= 0.0:
        raise ValueError("vmac must be positive.")

    bin_width = wavelength[1] - wavelength[0]
    vc = (vmac / c).decompose().value

    # np.convolve assumes the same kernel over the whole spectrum
    # However, the kernel depends on the wavelength (delta_lambda_L=lambda/c* vsini)
    # Solution: divide the spectrum into n_chunks
    # and convolve each chunk with its own kernel

    # Calculate kernel widths at chunk centers
    delta_lambda_L_array = wavelength * vc

    # total relative variation over full wavelength range
    rel_var_total = (
        np.max(delta_lambda_L_array) - np.min(delta_lambda_L_array)
    ) / np.mean(delta_lambda_L_array)

    # Estimate chunks assuming variation evenly split across chunks
    n_chunks = int(np.ceil(rel_var_total / epsilon))
    logger.debug(f"Divide spectrum into {n_chunks} chunks")

    # Define coarse wavelength grid for chunks
    wave_chunks = np.linspace(wavelength[0], wavelength[-1], n_chunks)

    # Global extension of wavelength and flux to avoid edge effects
    pad = int(np.ceil(vc * np.max(wavelength) / bin_width)) + 1
    if edge_handling == "firstlast":
        # extend the spectrum by padding it with the first and last values
        # makes sense for non-normalized spectra which will be extended by continuum values on edges
        flux_ext = np.concatenate([np.full(pad, flux[0]), flux, np.full(pad, flux[-1])])
        wave_ext = np.concatenate(
            [
                wavelength[0] + bin_width * np.arange(-pad, 0),
                wavelength,
                wavelength[-1] + bin_width * np.arange(1, pad + 1),
            ]
        )
    elif edge_handling == "normalized":
        # extend the spectrum by padding it with 1 on both sides
        # makes sense for normalized spectra which will be extended by continuum (1.0) values on edges
        flux_ext = np.concatenate([np.full(pad, 1.0), flux, np.full(pad, 1.0)])
        wave_ext = np.concatenate(
            [
                wavelength[0] + bin_width * np.arange(-pad, 0),
                wavelength,
                wavelength[-1] + bin_width * np.arange(1, pad + 1),
            ]
        )
    else:
        raise ValueError(f"Unsupported edge_handling: {edge_handling}")

    # Prepare output array - use normalized flux (1)
    broadened_flux = []

    # iterate over chunks
    for i in range(len(wave_chunks) - 1):
        L0, L1 = wave_chunks[i], wave_chunks[i + 1]
        lambda_mid = 0.5 * (L0 + L1)

        # maximum wavelength shift at midpoint of chunk
        delta_lambda_L = lambda_mid * vc
        ext_width = delta_lambda_L  # extend beyond chunk by ~kernel width

        chunk_mask = (wave_ext >= L0) & (wave_ext < L1)
        chunk_ext_mask = (wave_ext >= L0 - ext_width) & (wave_ext < L1 + ext_width)

        flux_chunk_ext = flux_ext[chunk_ext_mask]

        # Define convolution kernel
        N = int(np.floor(1.5 * delta_lambda_L / bin_width))
        delta_lambda = np.arange(-N, N + 1) * bin_width
        kernel = macroturbulence_kernel(delta_lambda_L, delta_lambda)

        # Convolve
        convolved_flux = np.convolve(flux_chunk_ext, kernel, mode="same")
        convolved_one = np.convolve(np.ones_like(flux_chunk_ext), kernel, mode="same")
        norm_flux = convolved_flux / convolved_one

        # Cut back to inner chunk region
        idx_in_chunk = np.where(chunk_mask & chunk_ext_mask)[0]
        idx_start = idx_in_chunk[0] - np.where(chunk_ext_mask)[0][0]
        idx_end = idx_in_chunk[-1] - np.where(chunk_ext_mask)[0][0] + 1
        broadened_flux.extend(norm_flux[idx_start:idx_end])

    broadened_flux = np.array(broadened_flux)
    if len(broadened_flux) < len(wavelength):
        logger.warning(
            f"WARNING: broadened_flux ({len(broadened_flux)}) is shorter than target length ({len(wavelength)}). Padding with last value."
        )
        broadened_flux = np.concatenate(
            [
                broadened_flux,
                np.full(len(wavelength) - len(broadened_flux), broadened_flux[-1]),
            ]
        )
    elif len(broadened_flux) > len(wavelength):
        logger.warning(
            f"WARNING: broadened_flux ({len(broadened_flux)}) is longer than target length ({len(wavelength)}). Trim excess values."
        )
        # Trim excess values
        broadened_flux = broadened_flux[: len(wavelength)]

    return broadened_flux
