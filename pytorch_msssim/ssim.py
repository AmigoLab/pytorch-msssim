# Copyright 2020 by Gongfan Fang, Zhejiang University.
# All rights reserved.
import warnings
from typing import Tuple, Union, Sequence, List
import numpy as np
import torch
import torch.nn.functional as F


def _fspecial_gauss_1d(size, sigma):
    r"""Create 1-D gauss kernel
    Args:
        size (int): the size of gauss kernel
        sigma (float): sigma of normal distribution

    Returns:
        torch.Tensor: 1D kernel (1 x 1 x size)
    """
    coords = torch.arange(size).to(dtype=torch.float)
    coords -= size // 2

    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()

    return g.unsqueeze(0).unsqueeze(0)


def _gaussian_filter(input, win):
    r""" Blur input with 1-D kernel
    Args:
        input (torch.Tensor): a batch of tensors to be blurred
        window (torch.Tensor): 1-D gauss kernel

    Returns:
        torch.Tensor: blurred tensors
    """
    assert all([ws == 1 for ws in win.shape[1:-1]]), win.shape
    if len(input.shape) == 4:
        conv = F.conv2d
    elif len(input.shape) == 5:
        conv = F.conv3d
    else:
        raise NotImplementedError(input.shape)

    C = input.shape[1]

    out = input

    for i, s in enumerate(input.shape[2:]):
        if s >= win.shape[-1]:
            out = conv(
                input=out,
                weight=win.transpose(2 + i, -1),
                padding=0,
                stride=1,
                groups=C,
                dilation=1,
            )
        else:
            warnings.warn(
                f"Skipping Gaussian Smoothing at dimension 2+{i} for input: {input.shape} and win size: {win.shape[-1]}"
            )

    return out


def _sobel_kernel(spatial_size: int) -> Tuple[torch.Tensor, ...]:
    """
        Constructs the sobel kernel for 2D or 3D kernel sizes.
        Args:
            spatial_size (int): The spatial size of the sobel kernel.
        Raises:
            ValueError when spatial_size is not 2 or 3.
    """
    if spatial_size == 2:
        g_x = (
            torch.Tensor([[+1, 0, -1], [+2, 0, -2], [+1, 0, -1]])
            .unsqueeze(0)
            .unsqueeze(0)
        )

        g_y = (
            torch.Tensor([[+1, +2, +1], [0, 0, 0], [-1, -2, 1]])
            .unsqueeze(0)
            .unsqueeze(0)
        )

        return g_x, g_y
    elif spatial_size == 3:
        g_x = (
            torch.Tensor(
                [
                    [[1, 0, -1], [1, 0, -1], [1, 0, -1]],
                    [[1, 0, -1], [2, 0, -2], [1, 0, -1]],
                    [[1, 0, -1], [1, 0, -1], [1, 0, -1]],
                ]
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )

        g_y = (
            torch.Tensor(
                [
                    [[1, 1, 1], [0, 0, 0], [-1, -1, -1]],
                    [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
                    [[1, 1, 1], [0, 0, 0], [-1, -1, -1]],
                ]
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )

        g_z = (
            torch.Tensor(
                [
                    [[1, 1, 1], [1, 2, 1], [1, 1, 1]],
                    [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                    [[-1, -1, -1], [-1, -2, -1], [-1, -1, -1]],
                ]
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )

        return g_x, g_y, g_z
    else:
        raise ValueError(
            f"Sobel operator is defined only for spatial sizes 2 and 3, but given {spatial_size}."
        )


def _gradient_map(
    input: torch.Tensor, gradient_kernels: List[torch.Tensor]
) -> torch.Tensor:
    """
    References:
        [1] Chen, G.H., Yang, C.L. and Xie, S.L., 2006, October.
        Gradient-based structural similarity for image quality assessment.
        In 2006 International Conference on Image Processing (pp. 2929-2932). IEEE.
    """
    gradient = torch.zeros_like(input)
    if len(input.shape) == 4:
        conv = F.conv2d
        gradient = gradient[:, :, 1:-1, 1:-1]
    elif len(input.shape) == 5:
        conv = F.conv3d
        gradient = gradient[:, :, 1:-1, 1:-1, 1:-1]
    else:
        raise NotImplementedError(input.shape)

    # We are following the gradient magnitude definition from Ref 1 - Section 3.1 - Eq 1
    for gk in gradient_kernels:
        directional_gradient = torch.abs(
            conv(
                input=input,
                weight=gk,
                stride=1,
                padding=0,
                groups=input.shape[1],
                dilation=1,
            )
        )
        gradient = gradient + directional_gradient

    return gradient


def _get_ssim_masks(x, y, no_masks, gradient_kernels):
    x_g = _gradient_map(input=x, gradient_kernels=gradient_kernels)
    y_g = _gradient_map(input=y, gradient_kernels=gradient_kernels)

    th1 = 0.12 * torch.max(x_g)
    th2 = 0.06 * torch.max(y_g)

    if no_masks == 3:
        m1 = torch.logical_and(x_g > th1, y_g > th1)
        m2 = torch.logical_and(x_g < th2, y_g <= th1)
        m3 = torch.logical_not(torch.logical_or(m1, m2))

        m1 = m1.float()
        m2 = m2.float()
        m3 = m3.float()

        return m1, m2, m3

    elif no_masks == 4:
        m1 = torch.logical_and(x_g > th1, y_g > th1)
        m2 = torch.logical_or(
            torch.logical_and(x_g > th1, y_g <= th1),
            torch.logical_and(x_g <= th1, y_g > th1),
        )
        m3 = torch.logical_and(x_g < th2, y_g > th1)
        m4 = torch.logical_not(torch.logical_or(torch.logical_or(m1, m2), m3))

        m1 = m1.float()
        m2 = m2.float()
        m3 = m3.float()
        m4 = m4.float()

        return m1, m2, m3, m4

    else:
        raise ValueError(
            f"Received no_masks ({no_masks}), but the accepted values are 3 and 4."
        )


def _ssim(
    x: torch.Tensor,
    y: torch.Tensor,
    data_range: float,
    win: torch.Tensor,
    k: Tuple[float, float] = (0.01, 0.03),
    scales: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    gradient_masks: Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ] = None,
    gradient_masks_weights: Union[
        Tuple[float, float, float], Tuple[float, float, float, float]
    ] = None,
    gradient_kernels: Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ] = None,
):

    """
    Calculate ssim index for X and Y

    Args:
        x (torch.Tensor): A batch of images, (N,C,[T,]H,W)
        y (torch.Tensor): A batch of images, (N,C,[T,]H,W)
        data_range (float): Maximum value of the input images assuming 0 is the minimum. Usually 1.0 or 255.0.
            Defaults to 255.0.
        gradient_masks (Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],]):
            Gradient masks that are determined based on the gradient magnitude of the images as per [3] or [4].
            Defaults to None.
        gradient_masks_weights (Union[Tuple[float, float, float], Tuple[float, float, float, float]]): The weight of the
            gradient masked regions. It also dictates the number of masks that will be used. It can be either 3 as per
            [3] or 4 as per [4]. If it is None no gradient masks will be applied. Defaults to None.

    Returns:
        torch.Tensor: ssim results.

    References:
        [1] Wang, Z., Bovik, A.C., Sheikh, H.R. and Simoncelli, E.P., 2004.
        Image quality assessment: from error visibility to structural similarity.
        IEEE transactions on image processing, 13(4), pp.600-612.

        [2] Chen, G.H., Yang, C.L. and Xie, S.L., 2006, October.
        Gradient-based structural similarity for image quality assessment.
        In 2006 International Conference on Image Processing (pp. 2929-2932). IEEE.

        [3] Li, C. and Bovik, A.C., 2009, January.
        Three-component weighted structural similarity index.
        In Image quality and system performance VI (Vol. 7242, p. 72420Q). International Society for Optics and Photonics.

        [4] Li, C. and Bovik, A.C., 2010.
        Content-partitioned structural similarity index for image quality assessment.
        Signal Processing: Image Communication, 25(7), pp.517-526.
    """
    K1, K2 = k
    alpha, beta, gamma = scales

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2
    C3 = C2 / 2

    win = win.to(x.device, dtype=x.dtype)

    # TODO: Replace this with fftconvolution
    mu1 = _gaussian_filter(x, win)
    mu2 = _gaussian_filter(y, win)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    # Ref 1 - Sec 3.B - Eq 6
    luminance = (2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)

    if gradient_kernels:
        if len(gradient_kernels) == 2:
            luminance = luminance[:, :, 1:-1, 1:-1]
        else:
            luminance = luminance[:, :, 1:-1, 1:-1, 1:-1]

        x = _gradient_map(input=x, gradient_kernels=gradient_kernels)
        y = _gradient_map(input=y, gradient_kernels=gradient_kernels)
        mu1 = _gaussian_filter(x, win)
        mu2 = _gaussian_filter(y, win)
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

    print(f"mu1 {torch.isnan(mu1).any()}")
    print(f"mu2 {torch.isnan(mu2).any()}")
    print(f"mu1_sq {torch.isnan(mu1_sq).any()}")
    print(f"mu2_sq {torch.isnan(mu2_sq).any()}")
    print(f"mu1_mu2 {torch.isnan(mu1_mu2).any()}")

    sigma1_sq = _gaussian_filter(x * x, win) - mu1_sq
    sigma2_sq = _gaussian_filter(y * y, win) - mu2_sq

    sigma12 = _gaussian_filter(x * y, win) - mu1_mu2
    sigma1 = torch.sqrt(sigma1_sq)
    sigma2 = torch.sqrt(sigma2_sq)
    print(f"sigma1_sq {torch.isnan(sigma1_sq).any()} {torch.min(sigma1_sq)}")
    print(f"sigma2_sq {torch.isnan(sigma2_sq).any()} {torch.min(sigma2_sq)}")
    print(f"sigma12 {torch.isnan(sigma12).any()}")
    print(f"sigma1 {torch.isnan(sigma1).any()}")
    print(f"sigma2 {torch.isnan(sigma2).any()}")
    # Ref 1 - Sec 3.B - Eq 9
    contrast = (2 * sigma1 * sigma2 + C2) / (sigma1_sq + sigma2_sq + C2)

    # Ref 1 - Sec 3.B - Eq 10
    structure = (sigma12 + C3) / (sigma1 * sigma2 + C3)
    print(f"luminance {torch.isnan(luminance).any()}")
    print(f"contrast {torch.isnan(contrast).any()}")
    print(f"structure {torch.isnan(structure).any()}")
    # Ref 1 - Sec 3.B - Eq 12
    luminance = torch.pow(luminance, alpha)
    contrast = torch.pow(contrast, beta)
    structure = torch.pow(structure, gamma)
    print(f"luminance {torch.isnan(luminance).any()}")
    print(f"contrast {torch.isnan(contrast).any()}")
    print(f"structure {torch.isnan(structure).any()}")
    ssim_map = luminance * contrast * structure

    if gradient_masks and gradient_masks_weights:
        ssim_per_channel = torch.zeros_like(
            torch.flatten(ssim_map, start_dim=2).mean(-1)
        )
        for gm, gmw in zip(gradient_masks, gradient_masks_weights):
            value = torch.flatten(input=ssim_map * gm, start_dim=2).sum(-1)
            value /= torch.flatten(input=gm, start_dim=2).sum(-1) + 1e-5
            value *= gmw
            ssim_per_channel += value
    else:
        ssim_per_channel = torch.flatten(ssim_map, start_dim=2).mean(-1)

    cs = torch.flatten(contrast, 2).mean(-1)

    return ssim_per_channel, cs


def ssim(
    x: torch.Tensor,
    y: torch.Tensor,
    data_range: float = 255.0,
    size_average: bool = True,
    win_size: int = 11,
    win_sigma: float = 1.5,
    win: torch.Tensor = None,
    k: Tuple[float, float] = (0.01, 0.03),
    scales: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    gradient_based: bool = False,
    gradient_masks_weights: Union[
        Tuple[float, float, float], Tuple[float, float, float, float]
    ] = None,
    nonnegative_ssim: bool = False,
):
    """
    Interface of Structural Similarity Index Metric

    Args:
        x (torch.Tensor): A batch of images, (N,C,[T,]H,W)
        y (torch.Tensor): A batch of images, (N,C,[T,]H,W)
        data_range (float): Maximum value of the input images assuming 0 is the minimum. Usually 1.0 or 255.0.
            Defaults to 255.0.
        TODO: Replace with reduction to fall in line with torch losses.
        size_average (bool): If size_average=True, SSIM of all images will be averaged as a scalar.
        win_size: (int): Size of the gaussian kernel. Defaults to 11 from Ref 1 - Sec 3.C.
        win_sigma: (float): Sigma of gaussian distribution. Defaults to 1.5 from Ref 1 - Sec 3.C.
        win (torch.Tensor): 1-D gaussian kernel. If None, a new kernel will be created according to win_size and
            win_sigma.
        k (list or tuple, optional): Scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a
            negative or NaN results. Defaults to (0.01, 0.03) from Ref 1 - Sec 3.C.
        scales (Tuple[float,float,float]): Scales of the luminance, contrast and structure components of the SSIM.
            Defaults to (1.0,1.0,1.0) from Ref 1 - Sec 3.B.
        gradient_based (bool): Whether or not the structural and contrast components of the SSIM are based on the
            gradient magnitudes of the images as per Ref 2. Defaults to False.
        gradient_masks_weights (Union[Tuple[float, float, float], Tuple[float, float, float, float]]): The weight of the
            gradient masked regions. It also dictates the number of masks that will be used. It can be either 3 as per
            [3] and the default values would be (0.5, 0.25, 0.25). Or 4 as per [4] and the default values would be
            (0.25, 0.25, 0.25, 0.25). If it is None no gradient masks will be applied. Defaults to None.
        nonnegative_ssim (bool): Whether or not the ssim results will be passed through a Sigmoid to guarantee they are
            [0,1]. Defaults to False.
    Returns:
        torch.Tensor: SSIM result

    References:
        [1] Wang, Z., Bovik, A.C., Sheikh, H.R. and Simoncelli, E.P., 2004.
        Image quality assessment: from error visibility to structural similarity.
        IEEE transactions on image processing, 13(4), pp.600-612.

        [2] Chen, G.H., Yang, C.L. and Xie, S.L., 2006, October.
        Gradient-based structural similarity for image quality assessment.
        In 2006 International Conference on Image Processing (pp. 2929-2932). IEEE.

        [3] Li, C. and Bovik, A.C., 2009, January.
        Three-component weighted structural similarity index.
        In Image quality and system performance VI (Vol. 7242, p. 72420Q). International Society for Optics and Photonics.

        [4] Li, C. and Bovik, A.C., 2010.
        Content-partitioned structural similarity index for image quality assessment.
        Signal Processing: Image Communication, 25(7), pp.517-526.
    """
    if not x.shape == y.shape:
        raise ValueError("Input images should have the same dimensions.")

    for d in range(len(x.shape) - 1, 1, -1):
        x = x.squeeze(dim=d)
        y = y.squeeze(dim=d)

    if len(x.shape) not in (4, 5):
        raise ValueError(
            f"Input images should be 4-d or 5-d tensors, but got {x.shape}"
        )

    if not x.type() == y.type():
        raise ValueError("Input images should have the same dtype.")

    if win is not None:  # set win_size
        win_size = win.shape[-1]

    if not (win_size % 2 == 1):
        raise ValueError("Window size should be odd.")

    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat([x.shape[1]] + [1] * (len(x.shape) - 1))

    # Here we calculate by how much the SSIM map differs from input image shape. This is calculated based on pytorch
    # conv documentation and the fact the stride is 1, padding is 0 and dilation is 1
    shape_delta = win_size - 1

    if gradient_based:
        gradient_kernels = _sobel_kernel(spatial_size=len(x.shape) - 2)
        gradient_kernels = [
            gk.repeat([x.shape[1]] + [1] * (len(x.shape) - 1))
            for gk in gradient_kernels
        ]
    else:
        gradient_kernels = None

    if gradient_masks_weights:
        gradient_kernels = _sobel_kernel(spatial_size=len(x.shape) - 2)
        gradient_kernels = [
            gk.repeat([x.shape[1]] + [1] * (len(x.shape) - 1))
            for gk in gradient_kernels
        ]

        gradient_masks = _get_ssim_masks(
            x=x,
            y=y,
            no_masks=len(gradient_masks_weights),
            gradient_kernels=gradient_kernels,
        )

        if len(x.shape) == 4:
            gradient_masks = tuple(
                gradient_mask[
                    :,
                    :,
                    shape_delta // 2 : -shape_delta // 2,
                    shape_delta // 2 : -shape_delta // 2,
                ]
                for gradient_mask in gradient_masks
            )
        else:
            gradient_masks = tuple(
                gradient_mask[
                    :,
                    :,
                    shape_delta // 2 : -shape_delta // 2,
                    shape_delta // 2 : -shape_delta // 2,
                    shape_delta // 2 : -shape_delta // 2,
                ]
                for gradient_mask in gradient_masks
            )
    else:
        gradient_masks = None

    ssim_per_channel, cs = _ssim(
        x,
        y,
        data_range=data_range,
        win=win,
        k=k,
        scales=scales,
        gradient_masks=gradient_masks,
        gradient_masks_weights=gradient_masks_weights,
        gradient_kernels=gradient_kernels,
    )

    if nonnegative_ssim:
        ssim_per_channel = torch.relu(ssim_per_channel)

    if size_average:
        return ssim_per_channel.mean()
    else:
        return ssim_per_channel.mean(1)


def ms_ssim(
    x: torch.Tensor,
    y: torch.Tensor,
    data_range: float = 255.0,
    size_average: bool = True,
    win_size: int = 11,
    win_sigma: float = 1.5,
    win: torch.Tensor = None,
    ms_weights: Tuple[float, ...] = None,
    k: Tuple[float, float] = (0.01, 0.03),
    scales: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    gradient_based: bool = False,
    gradient_masks_weights: Union[
        Tuple[float, float, float], Tuple[float, float, float, float]
    ] = None,
):
    """
    Interface of Multi Scale - Structural Similarity Index Metric

    Args:
        x (torch.Tensor): A batch of images, (N,C,[T,]H,W)
        y (torch.Tensor): A batch of images, (N,C,[T,]H,W)
        data_range (float): Maximum value of the input images assuming 0 is the minimum. Usually 1.0 or 255.0.
            Defaults to 255.0.
        TODO: Replace with reduction to fall in line with torch losses.
        size_average (bool): If size_average=True, SSIM of all images will be averaged as a scalar.
        win_size: (int): Size of the gaussian kernel. Defaults to 11 from Ref 1 - Sec 3.C.
        win_sigma: (float): Sigma of gaussian distribution. Defaults to 1.5 from Ref 1 - Sec 3.C.
        win (torch.Tensor): 1-D gaussian kernel. If None, a new kernel will be created according to win_size and
            win_sigma.
        ms_weights (Tuple[float,...]): Weights for different scales. It also dictates the number of the scales.
            Defaults to (0.0448, 0.2856, 0.3001, 0.2363, 0.1333) from Ref 2 - Sec 3.2 .
        k (list or tuple, optional): Scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a
            negative or NaN results. Defaults to (0.01, 0.03) from Ref 1 - Sec 3.C.
        scales (Tuple[float,float,float]): Scales of the luminance, contrast and structure components of the SSIM.
            Defaults to (1.0,1.0,1.0) from Ref 1 - Sec 3.B.
        gradient_based (bool): Whether or not the structural and contrast components of the SSIM are based on the
            gradient magnitudes of the images as per Ref 3. Defaults to False.
        gradient_masks_weights (Union[Tuple[float, float, float], Tuple[float, float, float, float]]): The weight of the
            gradient masked regions. It also dictates the number of masks that will be used. It can be either 3 as per
            [4] and the default values would be (0.5, 0.25, 0.25). Or 4 as per [5] and the default values would be
            (0.25, 0.25, 0.25, 0.25). If it is None no gradient masks will be applied. Defaults to None.

    Returns:
        torch.Tensor: MS-SSIM result

    References:
        [1] Wang, Z., Bovik, A.C., Sheikh, H.R. and Simoncelli, E.P., 2004.
        Image quality assessment: from error visibility to structural similarity.
        IEEE transactions on image processing, 13(4), pp.600-612.

        [2] Wang, Z., Simoncelli, E.P. and Bovik, A.C., 2003, November.
        Multiscale structural similarity for image quality assessment.
        In The Thrity-Seventh Asilomar Conference on Signals, Systems & Computers, 2003 (Vol. 2, pp. 1398-1402). Ieee.

        [3] Chen, G.H., Yang, C.L. and Xie, S.L., 2006, October.
        Gradient-based structural similarity for image quality assessment.
        In 2006 International Conference on Image Processing (pp. 2929-2932). IEEE.

        [4] Li, C. and Bovik, A.C., 2009, January.
        Three-component weighted structural similarity index.
        In Image quality and system performance VI (Vol. 7242, p. 72420Q). International Society for Optics and Photonics.

        [5] Li, C. and Bovik, A.C., 2010.
        Content-partitioned structural similarity index for image quality assessment.
        Signal Processing: Image Communication, 25(7), pp.517-526.
    """
    if not x.shape == y.shape:
        raise ValueError("Input images should have the same dimensions.")

    for d in range(len(x.shape) - 1, 1, -1):
        x = x.squeeze(dim=d)
        y = y.squeeze(dim=d)

    if not x.type() == y.type():
        raise ValueError("Input images should have the same dtype.")

    if len(x.shape) == 4:
        avg_pool = F.avg_pool2d
    elif len(x.shape) == 5:
        avg_pool = F.avg_pool3d
    else:
        raise ValueError(
            f"Input images should be 4-d or 5-d tensors, but got {x.shape}"
        )

    if win is not None:  # set win_size
        win_size = win.shape[-1]

    if not (win_size % 2 == 1):
        raise ValueError("Window size should be odd.")

    smaller_side = min(x.shape[-2:])
    assert smaller_side > (win_size - 1) * (2 ** 4), (
        "Image size should be larger than %d due to the 4 downsamplings in ms-ssim"
        % ((win_size - 1) * (2 ** 4))
    )

    if ms_weights is None:
        ms_weights = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)
    ms_weights = torch.FloatTensor(ms_weights).to(x.device, dtype=x.dtype)

    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat([x.shape[1]] + [1] * (len(x.shape) - 1))

    if gradient_based:
        gradient_kernels = _sobel_kernel(spatial_size=len(x.shape) - 2)
        gradient_kernels = [
            gk.repeat([x.shape[1]] + [1] * (len(x.shape) - 1))
            for gk in gradient_kernels
        ]
    else:
        gradient_kernels = None

    if gradient_masks_weights:
        gradient_kernels = _sobel_kernel(spatial_size=len(x.shape) - 2)
        gradient_kernels = [
            gk.repeat([x.shape[1]] + [1] * (len(x.shape) - 1))
            for gk in gradient_kernels
        ]

        gradient_masks = _get_ssim_masks(
            x=x,
            y=y,
            no_masks=len(gradient_masks_weights),
            gradient_kernels=gradient_kernels,
        )
    else:
        gradient_masks = None

    levels = ms_weights.shape[0]
    mcs = []
    for i in range(levels):
        ssim_per_channel, cs = _ssim(
            x,
            y,
            data_range=data_range,
            win=win,
            k=k,
            scales=scales,
            gradient_masks=gradient_masks,
            gradient_masks_weights=gradient_masks_weights,
            gradient_kernels=gradient_kernels,
        )

        if i < levels - 1:
            mcs.append(torch.relu(cs))
            padding = [s % 2 for s in x.shape[2:]]
            x = avg_pool(x, kernel_size=2, padding=padding)
            y = avg_pool(y, kernel_size=2, padding=padding)

    ssim_per_channel = torch.relu(ssim_per_channel)
    mcs_and_ssim = torch.stack(mcs + [ssim_per_channel], dim=0)
    ms_ssim_val = torch.prod(mcs_and_ssim ** ms_weights.view(-1, 1, 1), dim=0)

    if size_average:
        return ms_ssim_val.mean()
    else:
        return ms_ssim_val.mean(1)


class SSIM(torch.nn.Module):
    def __init__(
        self,
        data_range=255,
        size_average=True,
        win_size=11,
        win_sigma=1.5,
        channel=3,
        spatial_dims=2,
        K=(0.01, 0.03),
        scales=(1, 1, 1),
        gradient_based=False,
        nonnegative_ssim=False,
    ):
        r""" class for ssim
        Args:
            data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
            size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
            win_size: (int, optional): the size of gauss kernel
            win_sigma: (float, optional): sigma of normal distribution
            channel (int, optional): input channels (default: 3)
            K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
            nonnegative_ssim (bool, optional): force the ssim response to be nonnegative with relu.
        """

        super(SSIM, self).__init__()
        self.win_size = win_size
        self.win = _fspecial_gauss_1d(win_size, win_sigma).repeat(
            [channel, 1] + [1] * spatial_dims
        )
        self.size_average = size_average
        self.data_range = data_range
        self.K = K
        self.scales = scales
        self.gradient_based = gradient_based
        self.nonnegative_ssim = nonnegative_ssim

    def forward(self, X, Y):
        return ssim(
            X,
            Y,
            data_range=self.data_range,
            size_average=self.size_average,
            win=self.win,
            k=self.K,
            scales=self.scales,
            gradient_based=self.gradient_based,
            nonnegative_ssim=self.nonnegative_ssim,
        )


class MS_SSIM(torch.nn.Module):
    def __init__(
        self,
        data_range: float = 255.0,
        size_average: bool = True,
        win_size: int = 11,
        win_sigma: float = 1.5,
        win: torch.Tensor = None,
        ms_weights: Tuple[float, ...] = None,
        k: Tuple[float, float] = (0.01, 0.03),
        scales: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        gradient_based: bool = False,
        gradient_masks_weights: Union[
            Tuple[float, float, float], Tuple[float, float, float, float]
        ] = None,
    ):
        """
        Class for Multi Scale - Structural Similarity Index Metric

        Args:
            data_range (float): Maximum value of the input images assuming 0 is the minimum. Usually 1.0 or 255.0.
                Defaults to 255.0.
            TODO: Replace with reduction to fall in line with torch losses.
            size_average (bool): If size_average=True, SSIM of all images will be averaged as a scalar.
            win_size: (int): Size of the gaussian kernel. Defaults to 11 from Ref 1 - Sec 3.C.
            win_sigma: (float): Sigma of gaussian distribution. Defaults to 1.5 from Ref 1 - Sec 3.C.
            win (torch.Tensor): 1-D gaussian kernel. If None, a new kernel will be created according to win_size and
                win_sigma.
            ms_weights (Tuple[float,...]): Weights for different scales. It also dictates the number of the scales.
                Defaults to (0.0448, 0.2856, 0.3001, 0.2363, 0.1333) from Ref 2 - Sec 3.2 .
            k (list or tuple, optional): Scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a
                negative or NaN results. Defaults to (0.01, 0.03) from Ref 1 - Sec 3.C.
            scales (Tuple[float,float,float]): Scales of the luminance, contrast and structure components of the SSIM.
                Defaults to (1.0,1.0,1.0) from Ref 1 - Sec 3.B.
            gradient_based (bool): Whether or not the structural and contrast components of the SSIM are based on the
                gradient magnitudes of the images as per Ref 3. Defaults to False.
            gradient_masks_weights (Union[Tuple[float, float, float], Tuple[float, float, float, float]]): The weight of the
                gradient masked regions. It also dictates the number of masks that will be used. It can be either 3 as per
                [4] and the default values would be (0.5, 0.25, 0.25). Or 4 as per [5] and the default values would be
                (0.25, 0.25, 0.25, 0.25). If it is None no gradient masks will be applied. Defaults to None.

        Returns:
            torch.Tensor: MS-SSIM result

        References:
            [1] Wang, Z., Bovik, A.C., Sheikh, H.R. and Simoncelli, E.P., 2004.
            Image quality assessment: from error visibility to structural similarity.
            IEEE transactions on image processing, 13(4), pp.600-612.

            [2] Wang, Z., Simoncelli, E.P. and Bovik, A.C., 2003, November.
            Multiscale structural similarity for image quality assessment.
            In The Thrity-Seventh Asilomar Conference on Signals, Systems & Computers, 2003 (Vol. 2, pp. 1398-1402). Ieee.

            [3] Chen, G.H., Yang, C.L. and Xie, S.L., 2006, October.
            Gradient-based structural similarity for image quality assessment.
            In 2006 International Conference on Image Processing (pp. 2929-2932). IEEE.

            [4] Li, C. and Bovik, A.C., 2009, January.
            Three-component weighted structural similarity index.
            In Image quality and system performance VI (Vol. 7242, p. 72420Q). International Society for Optics and Photonics.

            [5] Li, C. and Bovik, A.C., 2010.
            Content-partitioned structural similarity index for image quality assessment.
            Signal Processing: Image Communication, 25(7), pp.517-526.
        """

        super(MS_SSIM, self).__init__()

        self.win_size = win_size
        self.win_sigma = win_sigma
        self.win = win
        self.size_average = size_average
        self.data_range = data_range
        self.ms_weights = ms_weights
        self.k = k
        self.scales = scales
        self.gradient_based = gradient_based
        self.gradient_masks_weights = gradient_masks_weights

    def forward(self, x, y):
        return ms_ssim(
            x,
            y,
            data_range=self.data_range,
            size_average=self.size_average,
            win_size=self.win_size,
            win_sigma=self.win_sigma,
            win=self.win,
            ms_weights=self.ms_weights,
            k=self.k,
            scales=self.scales,
            gradient_based=self.gradient_based,
        )
