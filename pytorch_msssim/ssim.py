# Copyright 2020 by Gongfan Fang, Zhejiang University.
# All rights reserved.
import warnings

import torch
import torch.nn.functional as F

from typing import Tuple


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


def gaussian_filter(input: torch.Tensor, win: torch.Tensor) -> torch.Tensor:
    """
    Blur input with 1-D kernel

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
    # TODO: Add choice of padding
    for i, s in enumerate(input.shape[2:]):
        if s >= win.shape[-1]:
            out = conv(
                out, weight=win.transpose(2 + i, -1), stride=1, padding=0, groups=C
            )
        else:
            warnings.warn(
                f"Skipping Gaussian Smoothing at dimension 2+{i} for input: {input.shape} and win size: {win.shape[-1]}"
            )

    return out


def _ssim_map(
    X: torch.Tensor,
    Y: torch.Tensor,
    data_range: float,
    win: torch.Tensor,
    K: Tuple[float, float] = (0.01, 0.03),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
        Given two tensors it calculates the resulting SSIM and contrast sensitivity maps.

        Args:
            X (torch.Tensor): images
            Y (torch.Tensor): images
            data_range (float): value range of input images.
            win (torch.Tensor): 1-D gauss kernel
            K (Tuple[float,float]): stability constants (K1, K2). Defaults to (0.01, 0.03).

        Returns:
            torch.Tensor: SSIM map
            torch.Tensor: contrast sensitivity map

        References:
            [1] Wang, Z., Bovik, A.C., Sheikh, H.R. and Simoncelli, E.P., 2004.
            Image quality assessment: from error visibility to structural similarity.
            IEEE transactions on image processing, 13(4), pp.600-612.
    """
    K1, K2 = K

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    win = win.to(X.device, dtype=X.dtype)

    #TODO: Replace this with fftconvolution
    mu1 = gaussian_filter(X, win)
    mu2 = gaussian_filter(Y, win)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = gaussian_filter(X * X, win) - mu1_sq
    sigma2_sq = gaussian_filter(Y * Y, win) - mu2_sq
    sigma12 = gaussian_filter(X * Y, win) - mu1_mu2

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)  # set alpha=beta=gamma=1
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map

    return ssim_map, cs_map


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
    ssim_map, cs_map = _ssim_map(X=X, Y=Y, data_range=data_range, win=win, K=K)
    # Getting the SSIM of each  image
    ssim_per_channel = torch.flatten(ssim_map, 2).mean(-1)
    cs = torch.flatten(cs_map, 2).mean(-1)
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

    # TODO: Understand why do we squeeze dimensions.
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
        ssim_per_channel, cs = _ssim(X, Y, win=win, data_range=data_range, K=K)

        if i < levels - 1:
            mcs.append(torch.relu(cs))
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
