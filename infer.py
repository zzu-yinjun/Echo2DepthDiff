# Last modified: 2024-05-24
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


import argparse
import logging
import os
from src.dataset.Beyound_Dataset import AudioVisualDataset
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from src.util.utils_criterion import compute_errors
from marigold import MarigoldPipeline
from src.util.seeding import seed_all
from src.dataset import (
    BaseDepthDataset,
    DatasetMode,
    get_dataset,
    get_pred_name,
)
def compute_scale_and_shift(prediction, target, mask):
    # 将预测值和目标值转换为列向量
    prediction = prediction[mask].reshape(-1)
    target = target[mask].reshape(-1)
    
    # 构建最小二乘问题的矩阵
    X = prediction.reshape(-1, 1)
    ones = np.ones_like(X)
    A = np.concatenate([X, ones], axis=1)
    
    # 使用最小二乘法求解
    scale, shift = np.linalg.lstsq(A, target, rcond=None)[0]
    
    return scale, shift
if "__main__" == __name__:
    logging.basicConfig(level=logging.INFO)

    # -------------------- Arguments --------------------
    parser = argparse.ArgumentParser(
        description="Run single-image depth estimation using Marigold."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="prs-eth/marigold-v1-0",
        help="Checkpoint path or hub name.",
    )

    # dataset setting
    parser.add_argument(
        "--dataset_config",
        type=str,
        required=True,
        help="Path to config file of evaluation dataset.",
    )
    parser.add_argument(
        "--base_data_dir",
        type=str,
        required=True,
        help="Path to base data directory.",
    )

    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory."
    )

    # inference setting
    parser.add_argument(
        "--denoise_steps",
        type=int,
        default=50,  # quantitative evaluation uses 50 steps
        help="Diffusion denoising steps, more steps results in higher accuracy but slower inference speed.",
    )
    parser.add_argument(
        "--ensemble_size",
        type=int,
        default=10,
        help="Number of predictions to be ensembled, more inference gives better results but runs slower.",
    )
    parser.add_argument(
        "--half_precision",
        "--fp16",
        action="store_true",
        help="Run with half-precision (16-bit float), might lead to suboptimal result.",
    )

    # resolution setting
    parser.add_argument(
        "--processing_res",
        type=int,
        default=0,
        help="Maximum resolution of processing. 0 for using input image resolution. Default: 0.",
    )
    parser.add_argument(
        "--output_processing_res",
        action="store_true",
        help="When input is resized, out put depth at resized operating resolution. Default: False.",
    )
    parser.add_argument(
        "--resample_method",
        type=str,
        default="bilinear",
        help="Resampling method used to resize images. This can be one of 'bilinear' or 'nearest'.",
    )

    parser.add_argument("--seed", type=int, default=None, help="Random seed.")

    args = parser.parse_args()

    checkpoint_path = args.checkpoint
    dataset_config = args.dataset_config
    base_data_dir = args.base_data_dir
    output_dir = args.output_dir

    denoise_steps = args.denoise_steps
    ensemble_size = args.ensemble_size
    if ensemble_size > 15:
        logging.warning("Running with large ensemble size will be slow.")
    half_precision = args.half_precision

    processing_res = args.processing_res
    match_input_res = not args.output_processing_res
    if 0 == processing_res and match_input_res is False:
        logging.warning(
            "Processing at native resolution without resizing output might NOT lead to exactly the same resolution, due to the padding and pooling properties of conv layers."
        )
    resample_method = args.resample_method

    seed = args.seed

    print(f"arguments: {args}")

    # -------------------- Preparation --------------------
    # Print out config
    logging.info(
        f"Inference settings: checkpoint = `{checkpoint_path}`, "
        f"with denoise_steps = {denoise_steps}, ensemble_size = {ensemble_size}, "
        f"processing resolution = {processing_res}, seed = {seed}; "
        f"dataset config = `{dataset_config}`."
    )

    # Random seed
    if seed is None:
        import time

        seed = int(time.time())
    seed_all(seed)

    def check_directory(directory):
        if os.path.exists(directory):
            response = (
                input(
                    f"The directory '{directory}' already exists. Are you sure to continue? (y/n): "
                )
                .strip()
                .lower()
            )
            if "y" == response:
                pass
            elif "n" == response:
                print("Exiting...")
                exit()
            else:
                print("Invalid input. Please enter 'y' (for Yes) or 'n' (for No).")
                check_directory(directory)  # Recursive call to ask again

    check_directory(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"output dir = {output_dir}")

    # -------------------- Device --------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        logging.warning("CUDA is not available. Running on CPU will be slow.")
    logging.info(f"device = {device}")

    # -------------------- Data --------------------
    cfg_data = OmegaConf.load(dataset_config)

    # dataset: BaseDepthDataset = get_dataset(
    #     cfg_data, base_data_dir=base_data_dir, mode=DatasetMode.RGB_ONLY
    # )

    # dataloader = DataLoader(dataset, batch_size=1, num_workers=0)
    # dataset = AudioVisualDataset('replica','test') 
    dataset = AudioVisualDataset('mp3d','test') 
    dataloader = DataLoader(dataset, batch_size =1,drop_last=True, shuffle=True, num_workers=4) 

    # -------------------- Model --------------------
    if half_precision:
        dtype = torch.float16
        variant = "fp16"
        logging.warning(
            f"Running with half precision ({dtype}), might lead to suboptimal result."
        )
    else:
        dtype = torch.float32
        variant = None

    pipe = MarigoldPipeline.from_pretrained(
        checkpoint_path, variant=variant, torch_dtype=dtype
    )

    try:
        pipe.enable_xformers_memory_efficient_attention()
    except ImportError:
        logging.debug("run without xformers")

    pipe = pipe.to(device)
    logging.info(
        f"scale_invariant: {pipe.scale_invariant}, shift_invariant: {pipe.shift_invariant}"
    )

    # -------------------- Inference and saving --------------------
    rmse_list = []
    abs_rel_list = []
    log10_list = []
    delta1_list = []
    delta2_list = []
    delta3_list = []
    mae_list = []
    errors = []
    with torch.no_grad():
          all_results = []  # 用于存储所有结果及其指标
          for i, batch in enumerate(tqdm(dataloader, leave=True)):

            

          
     
            # 从批次数据中获取RGB图像，并进行处理
            # rgb_int = batch["img"].squeeze().numpy().astype(np.uint8)  # 将图像数据转换为uint8类型的numpy数组，形状为[3, H, W]
            # rgb_int = np.moveaxis(rgb_int, 0, -1)  # 将通道维度从第一个位置移动到最后一个位置，形状变为[H, W, 3]
            # input_image = Image.fromarray(rgb_int)  # 将numpy数组转换为PIL图像对象
            rgb_int = batch["audio_spec"].squeeze().numpy().astype(np.uint8)  # 将图像数据转换为uint8类型的numpy数组，形状为[3, H, W]
        
            rgb_int = np.moveaxis(rgb_int, 0, -1)  # 将通道维度从第一个位置移动到最后一个位置，形状变为[H, W, 3]
            print(rgb_int.shape)    
            # input_image = Image.fromarray(rgb_int)  # 将numpy数组转换为PIL图像对象
            # rgb_int = batch['rgb_original'].squeeze().numpy().astype(np.uint8)  # 将图像数据转换为uint8类型的numpy数组，形状为[3, H, W]

            # print(rgb_int.shape)
            input_image = Image.fromarray(rgb_int)  # 将numpy数组转换为PIL图像对象
            audio_wave= batch["audio_wave"].to(device)
            #input_image (128,128,RGB)
            # Predict depth
            pipe_out = pipe(
                input_image,
                audio_input=audio_wave,
                denoising_steps=denoise_steps,
                ensemble_size=ensemble_size,
                processing_res=processing_res,
                match_input_res=match_input_res,
                batch_size=0,
                color_map=None,
                show_progress_bar=False,
                resample_method=resample_method,
               
            )
             
            



            # max_depth = 14.104
            depth_pred: np.ndarray = pipe_out.depth_np
            # print(depth_pred.shape) 128,128
   

            gt_depth = batch["depth_original"][0].cpu().numpy().squeeze(0)
            valid_mask = batch["valid_mask_raw"][0].cpu().numpy().squeeze(0)
            
            min_depth = 0.1
            max_depth = 14.0  # 与训练时使用的最大深度一致
            # max_depth = 14.1  # 与训练时使用的最大深度一致
# 
            # depth_pred = depth_pred * (max_depth - min_depth) + min_depth
            depth_pred = (depth_pred + 1.0) * (max_depth - min_depth) / 2.0 + min_depth

 
            save_dir = os.path.join(output_dir, "predictions")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            if valid_mask.sum() > 0:  # 确保有效区域非空
                # 计算中位数尺度因子
                scale = np.median(gt_depth[valid_mask]) / np.median(depth_pred[valid_mask])
                # 应用尺度对齐
                depth_pred_aligned = depth_pred * scale
                print(f"Applied median scale factor: {scale:.4f}")
                print(f"After alignment - Pred depth range: [{depth_pred_aligned.min():.3f}, {depth_pred_aligned.max():.3f}]")
                # scale, shift = compute_scale_and_shift(depth_pred, gt_depth, valid_mask)
                # # 应用尺度和偏移对齐
                # depth_pred_aligned = depth_pred * scale + shift
                # print(f"Applied least squares scale factor: {scale:.4f}, shift: {shift:.4f}")
                # print(f"After alignment - Pred depth range: [{depth_pred_aligned.min():.3f}, {depth_pred_aligned.max():.3f}]")
            else:
                depth_pred_aligned = depth_pred
                print("Warning: No valid pixels for scale alignment")
 
            depth_pred = depth_pred_aligned
            # 导入必要的库
            import matplotlib.pyplot as plt
            from matplotlib import cm
            import time
            import uuid

            # 生成唯一的文件名前缀
            unique_id = f"{int(time.time())}_{str(uuid.uuid4())[:8]}"

            # 获取数据
            rgb_int = batch["img"].squeeze().numpy()  # [3, H, W]
            rgb_int = np.moveaxis(rgb_int, 0, -1)  # [H, W, 3]
            # 将[-1,1]范围转换回[0,1]范围用于显示
            rgb_display = (rgb_int + 1.0) / 2.0

        
            # 创建一个3列的图表
            plt.figure(figsize=(18, 6))

            # 1. 显示原始RGB图像
            plt.subplot(131)
            plt.imshow(rgb_display)
            plt.title('RGB Image')
            plt.axis('off')

            # 2. 显示预测的深度图
            plt.subplot(132)
            depth_vis = plt.imshow(depth_pred_aligned, cmap='jet')
            plt.colorbar(depth_vis, label='Predicted Depth')
            plt.title('Predicted Depth')
            plt.axis('off')

            # 3. 显示真实深度图
            plt.subplot(133)
            # 使用掩码处理真实深度图，将无效区域设为NaN
            masked_gt_depth = gt_depth.copy()
            # masked_gt_depth[~valid_mask] = np.nan
            gt_vis = plt.imshow(masked_gt_depth, cmap='jet')
            plt.colorbar(gt_vis, label='Ground Truth Depth')
            plt.title('Ground Truth Depth')
            plt.axis('off')

            # 添加整体标题
            plt.suptitle(f'Depth Estimation Results', fontsize=16)
            plt.tight_layout()

            # 保存组合图像
            combined_save_path = os.path.join(save_dir, f"combined_{i:04d}.png")
            plt.savefig(combined_save_path, dpi=150)
            plt.close()

            # 可选：单独保存原始数据文件
            # depth_save_path = os.path.join(save_dir, f"depth_pred_{i:04d}.npy")
            # np.save(depth_save_path, depth_pred)

            print(f"Saved combined visualization to {combined_save_path}")
   

       
        
            

            # 使用对齐后的深度图计算指标
 
            abs_rel, rmse, a1, a2, a3, log_10, mae = compute_errors(
                gt_depth[valid_mask], depth_pred_aligned[valid_mask]
)
                 
            errors.append((abs_rel, rmse, a1, a2, a3, log_10, mae))

              # 然后在for循环内部的计算指标后添加以下代码:
            # (在计算errors.append之后)
            # 存储当前样本的结果信息
            sample_result = {
                'index': i,
                'rmse': rmse,
                'abs_rel': abs_rel, 
                'rgb_image': rgb_display.copy(),  # 原始RGB图像
                'pred_depth': depth_pred.copy(),  # 预测深度图
                'gt_depth': gt_depth.copy(),      # 真实深度图
                'valid_mask': valid_mask.copy()   # 有效区域掩码
            }
            all_results.append(sample_result)

            rmse_list.append(rmse)
            abs_rel_list.append(abs_rel)
            log10_list.append(log_10)
            delta1_list.append(a1)
            delta2_list.append(a2)
            delta3_list.append(a3)
            mae_list.append(mae)

          mean_errors = np.array(errors).mean(0)
    
          print("abs rel: {:.3f}".format(mean_errors[0]))
          print("RMSE: {:.3f}".format(mean_errors[1]))
          print("Delta1: {:.3f}".format(mean_errors[2]))
          print("Delta2: {:.3f}".format(mean_errors[3]))
          print("Delta3: {:.3f}".format(mean_errors[4]))
          print("Log10: {:.3f}".format(mean_errors[5]))
          print("MAE: {:.3f}".format(mean_errors[6]))

        #   best_results_dir = os.path.join(output_dir, "best_results")
        #   os.makedirs(best_results_dir, exist_ok=True)

        #     # 根据RMSE指标对结果排序（升序，即最小的RMSE在前）
        #   sorted_results = sorted(all_results, key=lambda x: x['rmse'])

        #     # 保存最好的5张深度图
        #   num_best = min(100, len(sorted_results))  # 确保我们有足够的样本
        #   print(f"\n保存最佳的{num_best}张深度图预测结果...")

        #   for rank, result in enumerate(sorted_results[:num_best]):
        #         idx = result['index']
        #         rmse = result['rmse']
        #         abs_rel = result['abs_rel']
                
        #         # 创建图表显示最佳结果
        #         plt.figure(figsize=(18, 6))
                
        #         # 1. 显示原始RGB图像
        #         plt.subplot(131)
        #         plt.imshow(result['rgb_image'])
        #         plt.title('RGB Image')
        #         plt.axis('off')
                
        #         # 2. 显示预测的深度图
        #         plt.subplot(132)
        #         depth_vis = plt.imshow(result['pred_depth'], cmap='jet')
        #         plt.colorbar(depth_vis, label='Predicted Depth')
        #         plt.title(f'Predicted Depth (RMSE: {rmse:.3f})')
        #         plt.axis('off')
                
        #         # 3. 显示真实深度图
        #         plt.subplot(133)
        #         # 使用掩码处理真实深度图
        #         masked_gt_depth = result['gt_depth'].copy()
        #         masked_gt_depth[~result['valid_mask']] = np.nan  # 将无效区域设为NaN
        #         gt_vis = plt.imshow(masked_gt_depth, cmap='jet')
        #         plt.colorbar(gt_vis, label='Ground Truth Depth')
        #         plt.title('Ground Truth Depth')
        #         plt.axis('off')
                
        #         # 添加整体标题
        #         plt.suptitle(f'Top {rank+1} Result (Sample #{idx}) - RMSE: {rmse:.3f}, Abs Rel: {abs_rel:.3f}', fontsize=16)
        #         plt.tight_layout()
                
        #         # 保存组合图像
        #         best_save_path = os.path.join(best_results_dir, f"best_{rank+1}_sample_{idx}_rmse_{rmse:.3f}.png")
        #         plt.savefig(best_save_path, dpi=300)
        #         plt.close()
                
        #         # 单独保存原始数据(可选)
        #         np.save(os.path.join(best_results_dir, f"best_{rank+1}_rgb_{idx}.npy"), result['rgb_image'])
        #         np.save(os.path.join(best_results_dir, f"best_{rank+1}_pred_{idx}.npy"), result['pred_depth'])
        #         np.save(os.path.join(best_results_dir, f"best_{rank+1}_gt_{idx}.npy"), result['gt_depth'])
                
        #         print(f"保存第{rank+1}佳结果 (样本 #{idx}): RMSE = {rmse:.3f}, Abs Rel = {abs_rel:.3f}")

          
 
            

        