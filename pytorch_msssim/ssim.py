# Copyright 2020 by Gongfan Fang, Zhejiang University.
# All rights reserved.
import warnings

import numpy as np
import torch
import torch.nn.functional as F

from typing import Tuple, Union, Sequence


def _common_checks(
    X: torch.tensor, Y: torch.tensor, win: torch.tensor, win_size: int, win_sigma: float
):
    if not X.shape == Y.shape:
        raise ValueError("Input images should have the same dimensions.")

    for d in range(len(X.shape) - 1, 1, -1):
        X = X.squeeze(dim=d)
        Y = Y.squeeze(dim=d)

    if len(X.shape) not in (4, 5):
        raise ValueError(
            f"Input images should be 4-d or 5-d tensors, but got {X.shape}"
        )

    if not X.type() == Y.type():
        raise ValueError("Input images should have the same dtype.")

    if win is not None:  # set win_size
        win_size = win.shape[-1]

    if not (win_size % 2 == 1):
        raise ValueError("Window size should be odd.")

    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat([X.shape[1]] + [1] * (len(X.shape) - 1))

    return X, Y, win


def same_padding(
    kernel_size: Union[Sequence[int], int], dilation: Union[Sequence[int], int] = 1
) -> Union[Tuple[int, ...], int]:
    """
    TAKEN FROM MONAI

    Return the padding value needed to ensure a convolution using the given kernel size produces an output of the same
    shape as the input for a stride of 1, otherwise ensure a shape of the input divided by the stride rounded down.

    Raises:
        NotImplementedError: When ``np.any((kernel_size - 1) * dilation % 2 == 1)``.

    """

    kernel_size_np = np.atleast_1d(kernel_size)
    dilation_np = np.atleast_1d(dilation)

    if np.any((kernel_size_np - 1) * dilation % 2 == 1):
        raise NotImplementedError(
            f"Same padding not available for kernel_size={kernel_size_np} and dilation={dilation_np}."
        )

    padding_np = (kernel_size_np - 1) / 2 * dilation_np
    padding = tuple(int(p) for p in padding_np)

    return padding if len(padding) > 1 else padding[0]


def _same_reflect_pad(input: torch.Tensor, kernel_size: int) -> torch.Tensor:
    spatial_dimensions = len(input.shape) - 2
    paddings = same_padding(kernel_size=kernel_size, dilation=1)

    if isinstance(paddings, int):
        paddings = [paddings] * spatial_dimensions

    # split_paddings = []
    # for p in paddings:
    #     split_paddings.extend([
    #         p // 2,
    #         p // 2 + p % 2
    #     ])

    return F.pad(input=input, pad=paddings, mode="replicate")


def _fspecial_gauss_1d(size: int, sigma: float) -> torch.Tensor:
    """
        Function to mimic the 'fspecial' gaussian MATLAB function.

        Args:
            size (int): the size of gauss kernel
            sigma (float): sigma of normal distribution

        Returns:
            torch.Tensor: 1D gaussian kernel (1 x 1 x size)
    """
    coords = torch.arange(size).to(dtype=torch.float)
    coords -= size // 2

    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()

    return g.unsqueeze(0).unsqueeze(0)


def _gaussian_filter(input: torch.Tensor, win: torch.Tensor) -> torch.Tensor:
    """
    Blur input with 1-D kernel

    Args:
        input (torch.Tensor): a batch of tensors to be blurred
        win (torch.Tensor): 1-D gaussian kernel
        same_pad (bool): Whether to pad the input to get a "SAME" convolution. This is needs to be True for advanced forms
            of MS-SSIM. Defaults to False.
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

    out = input
    for i, s in enumerate(out.shape[2:]):
        if s >= win.shape[-1]:
            weight = win.transpose(2 + i, -1)
            out = conv(
                out,
                weight=weight,
                stride=1,
                padding=same_padding(kernel_size=weight.shape[2:], dilation=1),
                groups=input.shape[1],
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


def _gradient_map(input: torch.Tensor) -> torch.Tensor:
    sobel_kernels = _sobel_kernel(len(input.shape) - 2)

    if len(input.shape) == 4:
        conv = F.conv2d
    elif len(input.shape) == 5:
        conv = F.conv3d
    else:
        raise NotImplementedError(input.shape)

    gradient = torch.zeros_like(input)

    # We are following the gradient magnitude definition from Section 3.1 - Eq 1
    # Chen, G.H., Yang, C.L. and Xie, S.L., 2006, October.
    # Gradient-based structural similarity for image quality assessment.
    # In 2006 International Conference on Image Processing (pp. 2929-2932). IEEE.
    for sk in sobel_kernels:
        directional_gradient = torch.abs(
            conv(
                input,
                weight=sk,
                stride=1,
                groups=input.shape[1],
                padding=same_padding(sk.shape[2:], dilation=1),
                dilation=1,
            )
        )
        gradient = gradient + directional_gradient

    return gradient


def _get_ssim_masks(X, Y, no_masks, same_pad):
    X_g = _gradient_map(input=X)
    Y_g = _gradient_map(input=Y)

    th1 = 0.12 * torch.max(X_g)
    th2 = 0.06 * torch.max(Y_g)

    if no_masks == 3:
        m1 = torch.logical_and(X_g > th1, Y_g > th1)
        m2 = torch.logical_and(X_g < th2, Y_g <= th1)
        m3 = torch.logical_not(torch.logical_or(m1, m2))

        m1 = m1.float()
        m2 = m2.float()
        m3 = m3.float()

        return m1, m2, m3

    elif no_masks == 4:
        m1 = torch.logical_and(X_g > th1, Y > th1)
        m2 = torch.logical_or(
            torch.logical_and(X_g > th1, Y_g <= th1),
            torch.logical_and(X_g <= th1, Y_g > th1),
        )
        m3 = torch.logical_and(X_g < th2, Y_g > th1)
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


def _ssim_map(
    X: torch.Tensor,
    Y: torch.Tensor,
    data_range: float,
    win: torch.Tensor,
    K: Tuple[float, float] = (0.01, 0.03),
    scales: Tuple[float, float, float] = (1, 1, 1),
    gradient_based: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
        Given two tensors it calculates the resulting SSIM and contrast sensitivity maps.

        Args:
            X (torch.Tensor): images
            Y (torch.Tensor): images
            data_range (float): value range of input images.
            win (torch.Tensor): 1-D gauss kernel
            K (Tuple[float,float]): stability constants (K1, K2). Defaults to (0.01, 0.03).
            gradient_based (bool): whether or not to use gradient based ssim.
        Returns:
            torch.Tensor: SSIM map
            torch.Tensor: contrast sensitivity map

        References:
            [1] Wang, Z., Bovik, A.C., Sheikh, H.R. and Simoncelli, E.P., 2004.
            Image quality assessment: from error visibility to structural similarity.
            IEEE transactions on image processing, 13(4), pp.600-612.
    """

    K1, K2 = K
    alpha, beta, gamma = scales

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2
    C3 = C2 / 2

    win = win.to(X.device, dtype=X.dtype)

    # TODO: Replace this with fftconvolution
    mu1 = _gaussian_filter(X, win)
    mu2 = _gaussian_filter(Y, win)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    print(f"mu1: {torch.isnan(mu1).any()}")
    print(f"mu2: {torch.isnan(mu2).any()}")
    print(f"mu1_sq: {torch.isnan(mu1_sq).any()}")
    print(f"mu2_sq: {torch.isnan(mu2_sq).any()}")
    print(f"mu1_mu2: {torch.isnan(mu1_mu2).any()}")

    # Ref 1 - Sec 3.B - Eq 6
    luminance = (2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)
    print(f"Luminance: {torch.isnan(luminance).any()}")
    if gradient_based:
        X = _gradient_map(input=X)
        Y = _gradient_map(input=Y)
        mu1 = _gaussian_filter(X, win)
        mu2 = _gaussian_filter(Y, win)
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

    # TODO: Understand why it is squared
    sigma1_sq = _gaussian_filter(X * X, win) - mu1_sq
    sigma2_sq = _gaussian_filter(Y * Y, win) - mu2_sq
    sigma12 = _gaussian_filter(X * Y, win) - mu1_mu2
    print(torch.min(sigma1_sq))
    print(torch.min(sigma2_sq))
    sigma1 = torch.sqrt(sigma1_sq)
    sigma2 = torch.sqrt(sigma2_sq)
    print(f"sigma1: {torch.isnan(sigma1).any()}")
    print(f"sigma2: {torch.isnan(sigma2).any()}")
    print(f"sigma12: {torch.isnan(sigma12).any()}")
    print(f"sigma1_sq: {torch.isnan(sigma1_sq).any()}")
    print(f"sigma2_sq: {torch.isnan(sigma2_sq).any()}")

    # Ref 1 - Sec 3.B - Eq 9
    contrast = (2 * sigma1 * sigma2 + C2) / (sigma1_sq + sigma2_sq + C2)
    print(f"Contrast: {torch.isnan(contrast).any()}")

    # Ref 1 - Sec 3.B - Eq 10
    structure = (sigma12 + C3) / (sigma1 * sigma2 + C3)
    print(f"Structure {torch.isnan(structure).any()}")

    # Ref 1 - Sec 3.B - Eq 12
    luminance = torch.pow(luminance, alpha)
    contrast = torch.pow(contrast, beta)
    structure = torch.pow(structure, gamma)

    ssim_map = luminance * contrast * structure

    return ssim_map, contrast


def _ssim(
    X: torch.Tensor,
    Y: torch.Tensor,
    data_range: float,
    win: torch.Tensor,
    K: Tuple[float, float] = (0.01, 0.03),
):
    """
        Calculate ssim index for X and Y

        Args:
            X (torch.Tensor): images
            Y (torch.Tensor): images
            data_range (float or int, optional): value range of input images.
            win (torch.Tensor): 1-D gauss kernel.
            K (Tuple[float,float]): stability constants (K1, K2). Defaults to (0.01, 0.03).

        Returns:
            torch.Tensor: ssim results
            torch.Tensor: contrast sensitivity

        References:
            [1] Wang, Z., Bovik, A.C., Sheikh, H.R. and Simoncelli, E.P., 2004.
            Image quality assessment: from error visibility to structural similarity.
            IEEE transactions on image processing, 13(4), pp.600-612.
    """
    # Calculating the SSIM maps
    ssim_map, cs_map = _ssim_map(
        X=X, Y=Y, data_range=data_range, win=win, K=K, gradient_based=False
    )
    print(torch.isnan(ssim_map).any())
    # Getting the SSIM of each  image
    ssim_per_channel = torch.flatten(ssim_map, 2).mean(-1)
    print(ssim_per_channel)
    cs = torch.flatten(cs_map, 2).mean(-1)
    print(cs)
    return ssim_per_channel, cs


def ssim(
    X: torch.Tensor,
    Y: torch.Tensor,
    data_range: float = 255,
    size_average: bool = True,
    win_size: int = 11,
    win_sigma: float = 1.5,
    win: torch.Tensor = None,
    K: Tuple[float, float] = (0.01, 0.03),
    strictly_positive_ssim: bool = False,
):
    """
        Functional form of the SSIM calculation.

        Args:
            X (torch.Tensor): a batch of images, (N,C,H,W)
            Y (torch.Tensor): a batch of images, (N,C,H,W)
            data_range (float or int, optional): value range of input images. Defaults to 255.
            size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar.
                Defaults to True.
            win_size: (int, optional): the size of gauss kernel. Defaults to 11.
            win_sigma: (float, optional): sigma of normal distribution. Defaults to 1.5
            win (torch.Tensor, optional): 1-D gauss kernel. If None, a new kernel will be created according to win_size
                and win_sigma. Defaults to None.
            K (Tuple[float,float]): scalar constants (K1, K2). Defaults to (0.01, 0.03)
            strictly_positive_ssim (bool, optional): force the ssim response to be strictly positive with relu.
                Defaults to False.

        Returns:
            torch.Tensor: SSIM scalar value

        References:
            [1] Wang, Z., Bovik, A.C., Sheikh, H.R. and Simoncelli, E.P., 2004.
            Image quality assessment: from error visibility to structural similarity.
            IEEE transactions on image processing, 13(4), pp.600-612.
    """
    if not X.shape == Y.shape:
        raise ValueError("Input images should have the same dimensions.")

    for d in range(len(X.shape) - 1, 1, -1):
        X = X.squeeze(dim=d)
        Y = Y.squeeze(dim=d)

    if len(X.shape) not in (4, 5):
        raise ValueError(
            f"Input images should be 4-d or 5-d tensors, but got {X.shape}"
        )

    if not X.type() == Y.type():
        raise ValueError("Input images should have the same dtype.")

    if win is not None:  # set win_size
        win_size = win.shape[-1]

    if not (win_size % 2 == 1):
        raise ValueError("Window size should be odd.")

    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat([X.shape[1]] + [1] * (len(X.shape) - 1))

    ssim_per_channel, cs = _ssim(X, Y, data_range=data_range, win=win, K=K)
    if strictly_positive_ssim:
        ssim_per_channel = torch.relu(ssim_per_channel)

    if size_average:
        return ssim_per_channel.mean()
    else:
        return ssim_per_channel.mean(1)


def ms_ssim(
    X: torch.Tensor,
    Y: torch.Tensor,
    data_range: float = 255,
    size_average: bool = True,
    win_size: int = 11,
    win_sigma: float = 1.5,
    win: torch.Tensor = None,
    weights: Tuple[float, ...] = None,
    K: Tuple[float, float] = (0.01, 0.03),
    gradient_based=False,
    wavelet_weights=None,
):
    """
        Functional form of ms-ssim

        Args:
            X (torch.Tensor): a batch of images, (N,C,[T,]H,W)
            Y (torch.Tensor): a batch of images, (N,C,[T,]H,W)
            data_range (float): value range of input images. (usually 1.0 or 255)
            size_average (bool): if size_average=True, ssim of all images will be averaged as a scalar
            win_size: (int): the size of gauss kernel
            win_sigma: (float): sigma of normal distribution
            win (torch.Tensor): 1-D gauss kernel. if None, a new kernel will be created according to win_size and win_sigma
            weights (Tuple[float,...]): weights for different levels
            K (Tuple[float,float]): scalar constants (K1, K2).

        Returns:
            torch.Tensor: MS-SSIM scalar value
    """
    if not X.shape == Y.shape:
        raise ValueError("Input images should have the same dimensions.")

    for d in range(len(X.shape) - 1, 1, -1):
        X = X.squeeze(dim=d)
        Y = Y.squeeze(dim=d)

    if not X.type() == Y.type():
        raise ValueError("Input images should have the same dtype.")

    if len(X.shape) == 4:
        avg_pool = F.avg_pool2d
    elif len(X.shape) == 5:
        avg_pool = F.avg_pool3d
    else:
        raise ValueError(
            f"Input images should be 4-d or 5-d tensors, but got {X.shape}"
        )

    if win is not None:  # set win_size
        win_size = win.shape[-1]

    if not (win_size % 2 == 1):
        raise ValueError("Window size should be odd.")

    smaller_side = min(X.shape[-2:])
    # TODO: Find out if this isn't wrong, shouldn't the power of 2 be correlated with the number of weights?
    assert smaller_side > (win_size - 1) * (2 ** 4), (
        "Image size should be larger than %d due to the 4 downsamplings in ms-ssim"
        % ((win_size - 1) * (2 ** 4))
    )

    if weights is None:
        weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    weights = torch.FloatTensor(weights).to(X.device, dtype=X.dtype)

    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat([X.shape[1]] + [1] * (len(X.shape) - 1))

    levels = weights.shape[0]
    mcs = []
    for i in range(levels):
        # TODO: Understand if this is right
        ssim_per_channel, cs = _ssim(
            X,
            Y,
            win=win,
            data_range=data_range,
            K=K,
            gradient_based=gradient_based,
            wavelet_weights=wavelet_weights,
        )

        if i < levels - 1:
            mcs.append(torch.relu(cs))
            # TODO: When symmetric padding is available replace the zero padding with symmetric padding.
            # TODO: Try reflection padding while we wait for symmetric padding.
            padding = [s % 2 for s in X.shape[2:]]
            X = avg_pool(X, kernel_size=2, padding=padding)
            Y = avg_pool(Y, kernel_size=2, padding=padding)

    ssim_per_channel = torch.relu(ssim_per_channel)  # (batch, channel)
    mcs_and_ssim = torch.stack(
        mcs + [ssim_per_channel], dim=0
    )  # (level, batch, channel)
    ms_ssim_val = torch.prod(mcs_and_ssim ** weights.view(-1, 1, 1), dim=0)

    if size_average:
        return ms_ssim_val.mean()
    else:
        return ms_ssim_val.mean(1)


# def component_ssim(
#     X: torch.Tensor,
#     Y: torch.Tensor,
#     data_range: float = 255,
#     size_average: bool = True,
#     win_size: int = 11,
#     win_sigma: float = 1.5,
#     win: torch.Tensor = None,
#     K: Tuple[float, float] = (0.01, 0.03),
#     keep_resolution: bool = False,
#     gradient_based: bool = False,
#     wavelet_weights: Tuple[float, ...] = None,
#     strictly_positive_ssim: bool = False,
# ):
#     if wavelet_weights is not None:
#         wavelet_masks = _get_ssim_masks(
#             X=X, Y=Y, no_masks=len(wavelet_weights), same_pad=keep_resolution
#         )
#
#     if not X.shape == Y.shape:
#         raise ValueError("Input images should have the same dimensions.")
#
#     for d in range(len(X.shape) - 1, 1, -1):
#         X = X.squeeze(dim=d)
#         Y = Y.squeeze(dim=d)
#
#     if len(X.shape) not in (4, 5):
#         raise ValueError(
#             f"Input images should be 4-d or 5-d tensors, but got {X.shape}"
#         )
#
#     if not X.type() == Y.type():
#         raise ValueError("Input images should have the same dtype.")
#
#     if win is not None:  # set win_size
#         win_size = win.shape[-1]
#
#     if not (win_size % 2 == 1):
#         raise ValueError("Window size should be odd.")
#
#     if win is None:
#         win = _fspecial_gauss_1d(win_size, win_sigma)
#         win = win.repeat([X.shape[1]] + [1] * (len(X.shape) - 1))
#
#     if wavelet_weights:
#         wavelet_ssim_map = torch.zeros_like(ssim_map)
#
#         for weight, mask in zip(wavelet_weights, wavelet_masks):
#             wavelet_ssim_map += weight * ssim_map * mask
#
#         ssim_map = wavelet_ssim_map


class SSIM(torch.nn.Module):
    """
        Class based SSIM.

        Args:
            data_range (float): value range of input images. Defaults to 255.
            size_average (bool): if size_average=True, ssim of all images will be averaged as a scalar.
                Defaults to True.
            win_size: (int): the size of gauss kernel. Defaults to 11.
            win_sigma: (float): sigma of normal distribution. Defaults to 1.5.
            channels (int): input channels. Defaults to 3.
            spatial_dims (int): number of spatial dimensions. Defaults to 2.
            K (Tuple[float,float]): scalar constants (K1, K2). Defaults to (0.01, 0.03).
            strictly_positive_ssim (bool): force the ssim response to be strictly positive with relu. Defaults to False.

        References:
            [1] Wang, Z., Bovik, A.C., Sheikh, H.R. and Simoncelli, E.P., 2004.
            Image quality assessment: from error visibility to structural similarity.
            IEEE transactions on image processing, 13(4), pp.600-612.
    """

    def __init__(
        self,
        data_range: float = 255,
        size_average: bool = True,
        win_size: int = 11,
        win_sigma: float = 1.5,
        channels: int = 3,
        spatial_dims: int = 2,
        K: Tuple[float, float] = (0.01, 0.03),
        strictly_positive_ssim=False,
    ):
        super(SSIM, self).__init__()
        self.win_size = win_size
        self.win = _fspecial_gauss_1d(win_size, win_sigma).repeat(
            [channels, 1] + [1] * spatial_dims
        )
        self.size_average = size_average
        self.data_range = data_range
        self.K = K
        self.strictly_positive_ssim = strictly_positive_ssim

    def forward(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
            Args:
                X (torch.Tensor): a batch of images, (N,C,[T,]H,W)
                Y (torch.Tensor): a batch of images, (N,C,[T,]H,W)

            Returns:
                torch.Tensor: SSIM scalar value
        """
        return ssim(
            X,
            Y,
            data_range=self.data_range,
            size_average=self.size_average,
            win=self.win,
            K=self.K,
            strictly_positive_ssim=self.strictly_positive_ssim,
        )


class MS_SSIM(torch.nn.Module):
    def __init__(
        self,
        data_range: float = 255,
        size_average: bool = True,
        win_size: int = 11,
        win_sigma: float = 1.5,
        channels: int = 3,
        spatial_dims: int = 2,
        weights: Tuple[float, ...] = None,
        K: Tuple[float, float] = (0.01, 0.03),
    ):
        """
        Class based MS-SSIM.

        Args:
            data_range (float): value range of input images. Defaults to 255.
            size_average (bool): if size_average=True, ssim of all images will be averaged as a scalar.
                Defaults to True.
            win_size: (int): the size of gauss kernel. Defaults to 11.
            win_sigma: (float): sigma of normal distribution. Defaults to 1.5.
            channels (int): input channels. Defaults to 3.
            spatial_dims (int): number of spatial dimensions. Defaults to 2.
            weights (list): weights for different levels. Defaults to None
            K (Tuple[float,float]): scalar constants (K1, K2). Defaults to (0.01, 0.03).

        References:
            [1] Wang, Z., Simoncelli, E.P. and Bovik, A.C., 2003, November.
            Multiscale structural similarity for image quality assessment.
            In The Thrity-Seventh Asilomar Conference on Signals, Systems & Computers, 2003 (Vol. 2, pp. 1398-1402). Ieee.
        """

        super(MS_SSIM, self).__init__()
        self.win_size = win_size
        self.win = _fspecial_gauss_1d(win_size, win_sigma).repeat(
            [channels, 1] + [1] * spatial_dims
        )
        self.size_average = size_average
        self.data_range = data_range
        self.weights = weights
        self.K = K

    def forward(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
            Args:
                X (torch.Tensor): a batch of images, (N,C,[T,]H,W)
                Y (torch.Tensor): a batch of images, (N,C,[T,]H,W)

            Returns:
                torch.Tensor: MS-SSIM scalar value
        """
        return ms_ssim(
            X,
            Y,
            data_range=self.data_range,
            size_average=self.size_average,
            win=self.win,
            weights=self.weights,
            K=self.K,
        )
