# Copyright 2023 Bingxin Ke, ETH Zurich. All rights reserved.
# Last modified: 2024-05-24
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


import logging
from typing import Dict, Optional, Union
import os
import numpy as np
import torch
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DiffusionPipeline,
    LCMScheduler,
    UNet2DConditionModel,
)
from diffusers.utils import BaseOutput
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import pil_to_tensor, resize
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from .util.batchsize import find_batch_size
from .util.ensemble import ensemble_depth
from .util.image_util import (
    chw2hwc,
    colorize_depth_maps,
    get_tv_resample_method,
    resize_max_res,
)
from .VAE_UNet import VAE_UNet_MultiASFF

class MarigoldDepthOutput(BaseOutput):
    """
    Output class for Marigold monocular depth prediction pipeline.

    Args:
        depth_np (`np.ndarray`):
            Predicted depth map, with depth values in the range of [0, 1].
        depth_colored (`PIL.Image.Image`):
            Colorized depth map, with the shape of [3, H, W] and values in [0, 1].
        uncertainty (`None` or `np.ndarray`):
            Uncalibrated uncertainty(MAD, median absolute deviation) coming from ensembling.
    """

    depth_np: np.ndarray
    depth_colored: Union[None, Image.Image]
    uncertainty: Union[None, np.ndarray]


class MarigoldPipeline(DiffusionPipeline):
    """
    Pipeline for monocular depth estimation using Marigold: https://marigoldmonodepth.github.io.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        unet (`UNet2DConditionModel`):
            Conditional U-Net to denoise the depth latent, conditioned on image latent.
        vae (`AutoencoderKL`):
            Variational Auto-Encoder (VAE) Model to encode and decode images and depth maps
            to and from latent representations.
        scheduler (`DDIMScheduler`):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents.
        text_encoder (`CLIPTextModel`):
            Text-encoder, for empty text embedding.
        tokenizer (`CLIPTokenizer`):
            CLIP tokenizer.
        scale_invariant (`bool`, *optional*):
            A model property specifying whether the predicted depth maps are scale-invariant. This value must be set in
            the model config. When used together with the `shift_invariant=True` flag, the model is also called
            "affine-invariant". NB: overriding this value is not supported.
        shift_invariant (`bool`, *optional*):
            A model property specifying whether the predicted depth maps are shift-invariant. This value must be set in
            the model config. When used together with the `scale_invariant=True` flag, the model is also called
            "affine-invariant". NB: overriding this value is not supported.
        default_denoising_steps (`int`, *optional*):
            The minimum number of denoising diffusion steps that are required to produce a prediction of reasonable
            quality with the given model. This value must be set in the model config. When the pipeline is called
            without explicitly setting `num_inference_steps`, the default value is used. This is required to ensure
            reasonable results with various model flavors compatible with the pipeline, such as those relying on very
            short denoising schedules (`LCMScheduler`) and those with full diffusion schedules (`DDIMScheduler`).
        default_processing_resolution (`int`, *optional*):
            The recommended value of the `processing_resolution` parameter of the pipeline. This value must be set in
            the model config. When the pipeline is called without explicitly setting `processing_resolution`, the
            default value is used. This is required to ensure reasonable results with various model flavors trained
            with varying optimal processing resolution values.
    """

    rgb_latent_scale_factor = 0.18215
    depth_latent_scale_factor = 0.18215

    def __init__(
        self,
        unet: UNet2DConditionModel,
        vae: AutoencoderKL,
        scheduler: Union[DDIMScheduler, LCMScheduler],
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        scale_invariant: Optional[bool] = True,
        shift_invariant: Optional[bool] = True,
        default_denoising_steps: Optional[int] = None,
        default_processing_resolution: Optional[int] = None,
        use_audio_embed: Optional[bool] = True,  # 新增参数
    ):
        super().__init__()
        self.register_modules(
            unet=unet,
            vae=vae,
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
        )
        self.register_to_config(
            scale_invariant=scale_invariant,
            shift_invariant=shift_invariant,
            default_denoising_steps=default_denoising_steps,
            default_processing_resolution=default_processing_resolution,
            use_audio_embed=use_audio_embed,
        )

        self.scale_invariant = scale_invariant
        self.shift_invariant = shift_invariant
        self.default_denoising_steps = default_denoising_steps
        self.default_processing_resolution = default_processing_resolution

        self.empty_text_embed = None
        
        if self.use_audio_embed:
            self.load_audio_embed_model()

        self.spec_vae = VAE_UNet_MultiASFF(
        in_channels=2,
        latent_channels=4,
        base_channels=32,
        num_downsample=3,
    )
        checkpoint = torch.load('/home/yinjun/project/Marigold-main/spec_vae/VAE_UNet_MultiASFF.pth')
        self.spec_vae.load_state_dict(checkpoint,strict=False)
        self.spec_vae = self.spec_vae.to(self.device)
        self.spec_vae.eval()

        ##########
    def load_audio_embed_model(self):
        """
        加载预训练的音频嵌入模型
        """
        import sys
        import os
        
        # 添加模型路径到Python路径
        model_path = os.path.dirname(os.path.abspath(__file__))
        if model_path not in sys.path:
            sys.path.append(model_path)
        
        # 导入模型
        from echo_depth import EchoDepthContrastiveModel  # 假设你的模型类名为AudioEmbedModel
        
        # 创建模型实例
        self.audio_embed_model = EchoDepthContrastiveModel()
        
        # 加载预训练权重
        # checkpoint_path = "/home/yinjun/project/Marigold-main/audio_depth_checkpoint/audio_depth_1.pth"
        checkpoint_path = "/home/yinjun/project/Marigold-main/output/mp3d/spec_wave_noPre/audio_embed_model.pth"
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        # 这里需要根据你的checkpoint格式调整
        if "model_state_dict" in checkpoint:
            self.audio_embed_model.load_state_dict(checkpoint["model_state_dict"])
        elif "state_dict" in checkpoint:
            self.audio_embed_model.load_state_dict(checkpoint["state_dict"])
        else:
            self.audio_embed_model.load_state_dict(checkpoint)
        
        # 移动到正确的设备并设置为评估模式
        self.audio_embed_model = self.audio_embed_model.to(self.device)
        self.audio_embed_model.eval()
        print("音频嵌入模型加载成功！")     

    @torch.no_grad()
    def __call__(
        self,
        input_image: Union[Image.Image, torch.Tensor],
        audio_input: Optional[torch.Tensor] = None,  # 新增音频输入参数
        denoising_steps: Optional[int] = None,
        ensemble_size: int = 5,
        processing_res: Optional[int] = None,
        match_input_res: bool = True,
        resample_method: str = "bilinear",
        batch_size: int = 0,
        generator: Union[torch.Generator, None] = None,
        color_map: str = "Spectral",
        show_progress_bar: bool = True,
        ensemble_kwargs: Dict = None,
    ) -> MarigoldDepthOutput:
        """
        Function invoked when calling the pipeline.

        Args:
            input_image (`Image`):
                Input RGB (or gray-scale) image.
            denoising_steps (`int`, *optional*, defaults to `None`):
                Number of denoising diffusion steps during inference. The default value `None` results in automatic
                selection. The number of steps should be at least 10 with the full Marigold models, and between 1 and 4
                for Marigold-LCM models.
            ensemble_size (`int`, *optional*, defaults to `10`):
                Number of predictions to be ensembled.
            processing_res (`int`, *optional*, defaults to `None`):
                Effective processing resolution. When set to `0`, processes at the original image resolution. This
                produces crisper predictions, but may also lead to the overall loss of global context. The default
                value `None` resolves to the optimal value from the model config.
            match_input_res (`bool`, *optional*, defaults to `True`):
                Resize depth prediction to match input resolution.
                Only valid if `processing_res` > 0.
            resample_method: (`str`, *optional*, defaults to `bilinear`):
                Resampling method used to resize images and depth predictions. This can be one of `bilinear`, `bicubic` or `nearest`, defaults to: `bilinear`.
            batch_size (`int`, *optional*, defaults to `0`):
                Inference batch size, no bigger than `num_ensemble`.
                If set to 0, the script will automatically decide the proper batch size.
            generator (`torch.Generator`, *optional*, defaults to `None`)
                Random generator for initial noise generation.
            show_progress_bar (`bool`, *optional*, defaults to `True`):
                Display a progress bar of diffusion denoising.
            color_map (`str`, *optional*, defaults to `"Spectral"`, pass `None` to skip colorized depth map generation):
                Colormap used to colorize the depth map.
            scale_invariant (`str`, *optional*, defaults to `True`):
                Flag of scale-invariant prediction, if True, scale will be adjusted from the raw prediction.
            shift_invariant (`str`, *optional*, defaults to `True`):
                Flag of shift-invariant prediction, if True, shift will be adjusted from the raw prediction, if False, near plane will be fixed at 0m.
            ensemble_kwargs (`dict`, *optional*, defaults to `None`):
                Arguments for detailed ensembling settings.
        Returns:
            `MarigoldDepthOutput`: Output class for Marigold monocular depth prediction pipeline, including:
            - **depth_np** (`np.ndarray`) Predicted depth map, with depth values in the range of [0, 1]
            - **depth_colored** (`PIL.Image.Image`) Colorized depth map, with the shape of [3, H, W] and values in [0, 1], None if `color_map` is `None`
            - **uncertainty** (`None` or `np.ndarray`) Uncalibrated uncertainty(MAD, median absolute deviation)
                    coming from ensembling. None if `ensemble_size = 1`
        """
        # Model-specific optimal default values leading to fast and reasonable results.
        if denoising_steps is None:
            denoising_steps = self.default_denoising_steps
        if processing_res is None:
            processing_res = self.default_processing_resolution

        assert processing_res >= 0
        assert ensemble_size >= 1

        # Check if denoising step is reasonable
        self._check_inference_step(denoising_steps)

        resample_method: InterpolationMode = get_tv_resample_method(resample_method)

        # ----------------- Image Preprocess -----------------
        # Convert to torch tensor
        if isinstance(input_image, Image.Image):
            input_image = input_image.convert("RGB")
            # convert to torch tensor [H, W, rgb] -> [rgb, H, W]
            rgb = pil_to_tensor(input_image)
            rgb = rgb.unsqueeze(0)  # [1, rgb, H, W]
        elif isinstance(input_image, torch.Tensor):
            rgb = input_image
        else:
            raise TypeError(f"Unknown input type: {type(input_image) = }")
        input_size = rgb.shape
        assert (
            4 == rgb.dim() and 3 == input_size[-3]
        ), f"Wrong input shape {input_size}, expected [1, rgb, H, W]"

        # Resize image
        if processing_res > 0:
            rgb = resize_max_res(
                rgb,
                max_edge_resolution=processing_res,
                resample_method=resample_method,
            )

        # Normalize rgb values
        # rgb_norm: torch.Tensor = rgb / 255.0 * 2.0 - 1.0  #  [0, 255] -> [-1, 1]
        rgb_norm=rgb
        rgb_norm = rgb_norm.to(self.dtype)
        assert rgb_norm.min() >= -1.0 and rgb_norm.max() <= 1.0
        
        # ----------------- Predicting depth -----------------
        # Batch repeated input image
        duplicated_rgb = rgb_norm.expand(ensemble_size, -1, -1, -1)
        single_rgb_dataset = TensorDataset(duplicated_rgb)
        if batch_size > 0:
            _bs = batch_size
        else:
            _bs = find_batch_size(
                ensemble_size=ensemble_size,
                input_res=max(rgb_norm.shape[1:]),
                dtype=self.dtype,
            )

        single_rgb_loader = DataLoader(
            single_rgb_dataset, batch_size=_bs, shuffle=False
        )

        # Predict depth maps (batched)
        depth_pred_ls = []
        if show_progress_bar:
            iterable = tqdm(
                single_rgb_loader, desc=" " * 2 + "Inference batches", leave=False
            )
        else:
            iterable = single_rgb_loader
        for batch in iterable:
            (batched_img,) = batch
            depth_pred_raw = self.single_infer(
                rgb_in=batched_img,
                num_inference_steps=denoising_steps,
                show_pbar=show_progress_bar,
                generator=generator,
                audio_in=audio_input
            )
        
          
            depth_pred_ls.append(depth_pred_raw.detach())
        depth_preds = torch.concat(depth_pred_ls, dim=0)
        # Output max and min values of depth predictions
  
        torch.cuda.empty_cache()  # clear vram cache for ensembling

 

        # ----------------- Test-time ensembling -----------------
        if ensemble_size > 1:
            depth_pred, pred_uncert = ensemble_depth(
                depth_preds,
                scale_invariant=self.scale_invariant,
                shift_invariant=self.shift_invariant,
                max_res=50,
                **(ensemble_kwargs or {}),
            )

        else:
            depth_pred = depth_preds
            pred_uncert = None

 
 
 
        # 如果需要匹配输入分辨率
        if match_input_res:
            # 将深度预测结果调整为输入图像的分辨率
            depth_pred = resize(
            depth_pred,
            input_size[-2:],  # 输入图像的高度和宽度
            interpolation=resample_method,  # 使用的重采样方法
            antialias=True,  # 是否使用抗锯齿
            )

        # Convert to numpy

        depth_pred = depth_pred.squeeze()
        depth_pred = depth_pred.cpu().numpy()
        if pred_uncert is not None:
            pred_uncert = pred_uncert.squeeze().cpu().numpy()

        # Clip output range
        depth_pred = depth_pred.clip(0, 1)

        # Colorize
        if color_map is not None:
            depth_colored = colorize_depth_maps(
                depth_pred, 0, 1, cmap=color_map
            ).squeeze()  # [3, H, W], value in (0, 1)
            depth_colored = (depth_colored * 255).astype(np.uint8)
            depth_colored_hwc = chw2hwc(depth_colored)
            depth_colored_img = Image.fromarray(depth_colored_hwc)
        else:
            depth_colored_img = None
        return MarigoldDepthOutput(
            depth_np=depth_pred,
            depth_colored=depth_colored_img,
            uncertainty=pred_uncert,
        )

    def _check_inference_step(self, n_step: int) -> None:
        """
        Check if denoising step is reasonable
        Args:
            n_step (`int`): denoising steps
        """
        assert n_step >= 1

        if isinstance(self.scheduler, DDIMScheduler):
            if n_step < 10:
                logging.warning(
                    f"Too few denoising steps: {n_step}. Recommended to use the LCM checkpoint for few-step inference."
                )
        elif isinstance(self.scheduler, LCMScheduler):
            if not 1 <= n_step <= 4:
                logging.warning(
                    f"Non-optimal setting of denoising steps: {n_step}. Recommended setting is 1-4 steps."
                )
        else:
            raise RuntimeError(f"Unsupported scheduler type: {type(self.scheduler)}")

    def encode_empty_text(self):
        """
        Encode text embedding for empty prompt
        """
        prompt = ""
        text_inputs = self.tokenizer(
            prompt,
            padding="do_not_pad",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.text_encoder.device)
        self.empty_text_embed = self.text_encoder(text_input_ids)[0].to(self.dtype)

    @torch.no_grad()
    def single_infer(
        self,
        rgb_in: torch.Tensor,
        num_inference_steps: int,
        generator: Union[torch.Generator, None],
        show_pbar: bool,
        audio_in: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Perform an individual depth prediction without ensembling.

        Args:
            rgb_in (`torch.Tensor`):
                Input RGB image.
            num_inference_steps (`int`):
                Number of diffusion denoisign steps (DDIM) during inference.
            show_pbar (`bool`):
                Display a progress bar of diffusion denoising.
            generator (`torch.Generator`)
                Random generator for initial noise generation.
        Returns:
            `torch.Tensor`: Predicted depth map.
        """
        device = self.device
        rgb_in = rgb_in.to(device)
        audio_in = audio_in.to(device) 

        # Set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps  # [T]

        # Encode image
        rgb_latent = self.encode_rgb(rgb_in)
    

        # Initial depth map (noise)
        depth_latent = torch.randn(
            rgb_latent.shape,
            device=device,
            dtype=self.dtype,
            generator=generator,
        )  # [B, 4, h, w]

       # 根据是否使用音频嵌入和是否提供音频输入来决定使用什么嵌入
  
        if self.use_audio_embed and audio_in is not None:
            # 使用音频嵌入模型生成嵌入
           
            batch_cond_embed = self.encode_audio(audio_in)
            batch_cond_embed = batch_cond_embed.repeat(
                (rgb_latent.shape[0], 1, 1) ).to(device)
           # [B, seq_len, hidden_dim]
        else:
            # 使用空文本嵌入
            if self.empty_text_embed is None:
                self.encode_empty_text()
            batch_cond_embed = self.empty_text_embed.repeat(
                (rgb_latent.shape[0], 1, 1)
            ).to(device)  # 
     
        # Denoising loop
        if show_pbar:
            iterable = tqdm(
                enumerate(timesteps),
                total=len(timesteps),
                leave=False,
                desc=" " * 4 + "Diffusion denoising",
            )
        else:
            iterable = enumerate(timesteps)

        for i, t in iterable:
            unet_input = torch.cat(
                [rgb_latent, depth_latent], dim=1
            )  # this order is important

            # predict the noise residual
            #batch_cond_embed ([10, 2, 1024])
            noise_pred = self.unet(
                unet_input, t, encoder_hidden_states=batch_cond_embed
            ).sample  # [B, 4, h, w]

            # compute the previous noisy sample x_t -> x_t-1
            depth_latent = self.scheduler.step(
                noise_pred, t, depth_latent, generator=generator
            ).prev_sample

        depth = self.decode_depth(depth_latent)

        # clip prediction
        depth = torch.clip(depth, -1.0, 1.0)
    
        # shift to [0, 1]
        # depth = (depth + 1.0) / 2.0
 
    
        return depth
    
    def encode_rgb(self, rgb_in: torch.Tensor) -> torch.Tensor:
     
        """
        Encode RGB image into latent.

        Args:
            rgb_in (`torch.Tensor`):
                Input RGB image to be encoded.

        Returns:
            `torch.Tensor`: Image latent.
        """
       
        # encode
        h = self.vae.encoder(rgb_in)
        moments = self.vae.quant_conv(h)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        # scale latent
        rgb_latent = mean * self.rgb_latent_scale_factor
 
        # print(f"rgb_latent shape: {rgb_latent.shape}") 8 4 16 16
 
        return rgb_latent
    def encode_spec(self, rgb_in: torch.Tensor) -> torch.Tensor:
     
        """
        Encode RGB image into latent.

        Args:
            rgb_in (`torch.Tensor`):
                Input RGB image to be encoded.

        Returns:
            `torch.Tensor`: Image latent.
        """
       
        # encode
        # h = self.vae.encoder(rgb_in)
        # moments = self.vae.quant_conv(h)
        # mean, logvar = torch.chunk(moments, 2, dim=1)
        # # scale latent
        # rgb_latent = mean * self.rgb_latent_scale_factor
 
        # print(f"rgb_latent shape: {rgb_latent.shape}") 8 4 16 16
        with torch.no_grad():
            # 使用VAE_UNet_MultiASFF编码
            self.spec_vae = self.spec_vae.to(self.device)
            rgb_in = rgb_in.to(self.device)
            encoded_output = self.spec_vae.encoder(rgb_in)

        if isinstance(encoded_output, tuple):
            rgb_latent = encoded_output[0]
        else:
            rgb_latent = encoded_output
        
        return rgb_latent



    def decode_depth(self, depth_latent: torch.Tensor) -> torch.Tensor:
        """
        Decode depth latent into depth map.

        Args:
            depth_latent (`torch.Tensor`):
                Depth latent to be decoded.

        Returns:
            `torch.Tensor`: Decoded depth map.
        """
        # scale latent
        depth_latent = depth_latent / self.depth_latent_scale_factor
        # decode
        z = self.vae.post_quant_conv(depth_latent)
        stacked = self.vae.decoder(z)
        # mean of output channels
        depth_mean = stacked.mean(dim=1, keepdim=True)

        return depth_mean
       
    def encode_audio(self, audio_in: torch.Tensor) -> torch.Tensor:
        """
        编码音频为条件嵌入向量
        
        Args:
            audio_in (torch.Tensor): 输入音频波形，形状为 [B, 2, T]
            
        Returns:
            torch.Tensor: 条件嵌入向量，形状为 [B, 77, 1024]
        """
        device = self.device
        
        with torch.no_grad():
            # 确保模型和输入在正确的设备上
            self.audio_embed_model = self.audio_embed_model.to(device)
            audio_in = audio_in.to(device)
            
            # 使用回声编码器生成嵌入
            audio_embed = self.audio_embed_model.echo_encoder(audio_in)
            
            # 添加序列维度如果需要
            if audio_embed.dim() == 2:  # [B, embed_dim]
                audio_embed = audio_embed.unsqueeze(1)  # [B, 1, embed_dim]
            
            # 添加投影层进行维度调整 (768 -> 1024)
            if not hasattr(self, 'audio_projection'):
                self.audio_projection = torch.nn.Sequential(
                    torch.nn.Linear(768, 1024),
                    torch.nn.LayerNorm(1024),
                    torch.nn.GELU(),
                ).to(device)
            projection_path = "/home/yinjun/project/Marigold-main/output/mp3d/spec_wave_noPre/audio_projection.pth"
            if os.path.exists(projection_path):
                self.audio_projection.load_state_dict(
                    torch.load(projection_path, map_location=device)
                )
                print(f"已加载音频投影层权重: {projection_path}")
            else:
                print(f"警告：未找到音频投影层权重文件: {projection_path}")
            
            self.audio_projection = self.audio_projection.to(device)
            
            # 进行维度投影
            audio_embed = self.audio_projection(audio_embed)  # [B, 1, 1024]
            
            # 将序列长度扩展到与CLIP文本编码器相同 (77个token)
            audio_embed = audio_embed.repeat(1, 77, 1)  # [B, 77, 1024]
            
            # 确保输出在正确的设备上
            audio_embed = audio_embed.to(device)
            
         
        
        return audio_embed