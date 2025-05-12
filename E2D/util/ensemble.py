# Copyright 2023 Bingxin Ke, ETH Zurich. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------------------------
# If you find this code useful, we kindly ask you to cite our paper in your work.
# Please find bibtex at: https://github.com/prs-eth/Marigold#-citation
# More information about the method can be found at https://marigoldmonodepth.github.io
# --------------------------------------------------------------------------


from functools import partial
from typing import Optional, Tuple

import numpy as np
import torch

from .image_util import get_tv_resample_method, resize_max_res


def inter_distances(tensors: torch.Tensor):
    """
    To calculate the distance between each two depth maps.
    """
    distances = []
    for i, j in torch.combinations(torch.arange(tensors.shape[0])):
        arr1 = tensors[i : i + 1]
        arr2 = tensors[j : j + 1]
        distances.append(arr1 - arr2)
    dist = torch.concatenate(distances, dim=0)
    return dist


def ensemble_depth(
    depth: torch.Tensor,
    scale_invariant: bool = True,
    shift_invariant: bool = True,
    output_uncertainty: bool = False,
    reduction: str = "median",
    regularizer_strength: float = 0.02,
    max_iter: int = 2,
    tol: float = 1e-3,
    max_res: int = 1024,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Ensembles depth maps represented by the `depth` tensor with expected shape `(B, 1, H, W)`, where B is the
    number of ensemble members for a given prediction of size `(H x W)`. Even though the function is designed for
    depth maps, it can also be used with disparity maps as long as the input tensor values are non-negative. The
    alignment happens when the predictions have one or more degrees of freedom, that is when they are either
    affine-invariant (`scale_invariant=True` and `shift_invariant=True`), or just scale-invariant (only
    `scale_invariant=True`). For absolute predictions (`scale_invariant=False` and `shift_invariant=False`)
    alignment is skipped and only ensembling is performed.

    Args:
        depth (`torch.Tensor`):
            Input ensemble depth maps.
        scale_invariant (`bool`, *optional*, defaults to `True`):
            Whether to treat predictions as scale-invariant.
        shift_invariant (`bool`, *optional*, defaults to `True`):
            Whether to treat predictions as shift-invariant.
        output_uncertainty (`bool`, *optional*, defaults to `False`):
            Whether to output uncertainty map.
        reduction (`str`, *optional*, defaults to `"median"`):
            Reduction method used to ensemble aligned predictions. The accepted values are: `"mean"` and
            `"median"`.
        regularizer_strength (`float`, *optional*, defaults to `0.02`):
            Strength of the regularizer that pulls the aligned predictions to the unit range from 0 to 1.
        max_iter (`int`, *optional*, defaults to `2`):
            Maximum number of the alignment solver steps. Refer to `scipy.optimize.minimize` function, `options`
            argument.
        tol (`float`, *optional*, defaults to `1e-3`):
            Alignment solver tolerance. The solver stops when the tolerance is reached.
        max_res (`int`, *optional*, defaults to `1024`):
            Resolution at which the alignment is performed; `None` matches the `processing_resolution`.
    Returns:
        A tensor of aligned and ensembled depth maps and optionally a tensor of uncertainties of the same shape:
        `(1, 1, H, W)`.
    """
    # 检查输入的深度张量是否为4维且第二维度为1
    if depth.dim() != 4 or depth.shape[1] != 1:
        raise ValueError(f"Expecting 4D tensor of shape [B,1,H,W]; got {depth.shape}.")
    # 检查reduction参数是否为"mean"或"median"
    if reduction not in ("mean", "median"):
        raise ValueError(f"Unrecognized reduction method: {reduction}.")
    # 检查是否为纯平移不变的情况，这种情况不支持
    if not scale_invariant and shift_invariant:
        raise ValueError("Pure shift-invariant ensembling is not supported.")

    # 初始化参数函数
    def init_param(depth: torch.Tensor):
        # 计算每个深度图的最小值和最大值
        init_min = depth.reshape(ensemble_size, -1).min(dim=1).values
        init_max = depth.reshape(ensemble_size, -1).max(dim=1).values

        # 如果是比例不变且平移不变的情况
        if scale_invariant and shift_invariant:
            init_s = 1.0 / (init_max - init_min).clamp(min=1e-6)  # 计算比例参数
            init_t = -init_s * init_min  # 计算平移参数
            param = torch.cat((init_s, init_t)).cpu().numpy()  # 合并参数
        # 如果仅是比例不变的情况
        elif scale_invariant:
            init_s = 1.0 / init_max.clamp(min=1e-6)  # 计算比例参数
            param = init_s.cpu().numpy()  # 转换为numpy数组
        else:
            raise ValueError("Unrecognized alignment.")  # 抛出未识别的对齐方式错误

        return param  # 返回初始化参数

    def align(depth: torch.Tensor, param: np.ndarray) -> torch.Tensor:
        if scale_invariant and shift_invariant:
            s, t = np.split(param, 2)
            s = torch.from_numpy(s).to(depth).view(ensemble_size, 1, 1, 1)
            t = torch.from_numpy(t).to(depth).view(ensemble_size, 1, 1, 1)
            out = depth * s + t
        elif scale_invariant:
            s = torch.from_numpy(param).to(depth).view(ensemble_size, 1, 1, 1)
            out = depth * s
        else:
            raise ValueError("Unrecognized alignment.")
        return out

    def ensemble(
        # """
        # 对齐深度图的集成方法。

        # 参数:
        # depth_aligned (torch.Tensor): 对齐的深度图张量，形状为 (N, H, W)，其中 N 是样本数量，H 和 W 是图像的高度和宽度。
        # return_uncertainty (bool): 是否返回不确定性估计。默认为 False。

        # 返回:
        # Tuple[torch.Tensor, Optional[torch.Tensor]]: 返回一个包含预测结果和不确定性估计的元组。
        #     - prediction (torch.Tensor): 预测结果张量，形状为 (1, H, W)。
        #     - uncertainty (Optional[torch.Tensor]): 不确定性估计张量，形状为 (1, H, W)，如果 return_uncertainty 为 False，则为 None。

        # 异常:
        # ValueError: 如果 reduction 方法未被识别，则抛出此异常。

        # 说明:
        # 该函数根据指定的 reduction 方法对对齐的深度图进行集成。支持的 reduction 方法包括 "mean" 和 "median"。
        # 如果 reduction 为 "mean"，则计算深度图的均值作为预测结果，并可选地计算标准差作为不确定性估计。
        # 如果 reduction 为 "median"，则计算深度图的中位数作为预测结果，并可选地计算绝对偏差的中位数作为不确定性估计。
        # """
        depth_aligned: torch.Tensor, return_uncertainty: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        uncertainty = None
        if reduction == "mean":
            prediction = torch.mean(depth_aligned, dim=0, keepdim=True)
            if return_uncertainty:
                uncertainty = torch.std(depth_aligned, dim=0, keepdim=True)
        elif reduction == "median":
            prediction = torch.median(depth_aligned, dim=0, keepdim=True).values
            if return_uncertainty:
                uncertainty = torch.median(
                    torch.abs(depth_aligned - prediction), dim=0, keepdim=True
                ).values
        else:
            raise ValueError(f"Unrecognized reduction method: {reduction}.")
        return prediction, uncertainty

    def cost_fn(param: np.ndarray, depth: torch.Tensor) -> float:
        cost = 0.0
        depth_aligned = align(depth, param)

        for i, j in torch.combinations(torch.arange(ensemble_size)):
            diff = depth_aligned[i] - depth_aligned[j]
            cost += (diff**2).mean().sqrt().item()

        if regularizer_strength > 0:
            prediction, _ = ensemble(depth_aligned, return_uncertainty=False)
            err_near = (0.0 - prediction.min()).abs().item()
            err_far = (1.0 - prediction.max()).abs().item()
            cost += (err_near + err_far) * regularizer_strength

        return cost

    def compute_param(depth: torch.Tensor):
        import scipy

        depth_to_align = depth.to(torch.float32)
        if max_res is not None and max(depth_to_align.shape[2:]) > max_res:
            depth_to_align = resize_max_res(
                depth_to_align, max_res, get_tv_resample_method("nearest-exact")
            )

        param = init_param(depth_to_align)

        res = scipy.optimize.minimize(
            partial(cost_fn, depth=depth_to_align),
            param,
            method="BFGS",
            tol=tol,
            options={"maxiter": max_iter, "disp": False},
        )

        return res.x

    requires_aligning = scale_invariant or shift_invariant
    ensemble_size = depth.shape[0]

    if requires_aligning:
        param = compute_param(depth)
        depth = align(depth, param)

    depth, uncertainty = ensemble(depth, return_uncertainty=output_uncertainty)

    depth_max = depth.max()
    if scale_invariant and shift_invariant:
        depth_min = depth.min()
    elif scale_invariant:
        depth_min = 0
    else:
        raise ValueError("Unrecognized alignment.")
    depth_range = (depth_max - depth_min).clamp(min=1e-6)
    depth = (depth - depth_min) / depth_range
    if output_uncertainty:
        uncertainty /= depth_range

    return depth, uncertainty  # [1,1,H,W], [1,1,H,W]
