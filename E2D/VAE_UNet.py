import torch
import torch.nn as nn
import torch.nn.functional as F

class Down(nn.Module):
    """带残差连接的下采样块"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.InstanceNorm2d(out_channels)
        )
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=2),
            nn.InstanceNorm2d(out_channels)
        )

    def forward(self, x):
        return self.main(x) + self.residual(x)

class ASPP(nn.Module):
    """空洞空间金字塔池化模块"""
    def __init__(self, in_channels, out_channels, dilation_rates=[1, 3, 5]):
        super().__init__()
        modules = []
        for rate in dilation_rates:
            modules.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate),
                nn.InstanceNorm2d(out_channels),
                nn.LeakyReLU(0.2, inplace=True)
            ))
        self.convs = nn.ModuleList(modules)
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.fusion = nn.Conv2d(out_channels * (len(dilation_rates) + 1), out_channels, 1)

    def forward(self, x):
        h, w = x.shape[2:]
        branch_outs = [conv(x) for conv in self.convs]
        global_out = self.global_pool(x)
        global_out = F.interpolate(global_out, size=(h, w), mode='bilinear', align_corners=False)
        return self.fusion(torch.cat(branch_outs + [global_out], dim=1))

class ASFF(nn.Module):
    """自适应空间特征融合模块（动态通道适配）"""
    def __init__(self, level, feature_channels):
        super().__init__()
        self.level = level
        self.feature_channels = feature_channels
        self.fusion = nn.Sequential(
            nn.Conv2d(sum(feature_channels), feature_channels[self.level], 1),
            nn.InstanceNorm2d(feature_channels[self.level]),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, *features):
        resized_features = []
        target_size = features[self.level].shape[2:]
        for feat in features:
            if feat.shape[2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            resized_features.append(feat)
        return self.fusion(torch.cat(resized_features, dim=1))

class TripleASFF(nn.Module):
    """三层自适应特征融合（精确通道控制）"""
    def __init__(self, feature_channels):
        super().__init__()
        self.asff_high = ASFF(level=0, feature_channels=feature_channels)
        self.asff_mid = ASFF(level=1, feature_channels=feature_channels)
        self.asff_low = ASFF(level=2, feature_channels=feature_channels)
        self.final_fusion = nn.Sequential(
            nn.Conv2d(sum(feature_channels), feature_channels[-1], 3, padding=1),
            nn.InstanceNorm2d(feature_channels[-1]),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, features):
        high, mid, low = features
        fused_high = self.asff_high(high, mid, low)
        fused_mid = self.asff_mid(high, mid, low)
        fused_low = self.asff_low(high, mid, low)
        
        # 统一到最低层分辨率
        target_size = low.shape[2:]
        fused_high = F.interpolate(fused_high, size=target_size, mode='bilinear')
        fused_mid = F.interpolate(fused_mid, size=target_size, mode='bilinear')
        
        return self.final_fusion(torch.cat([fused_high, fused_mid, fused_low], dim=1))

class VAE_Encoder(nn.Module):
    """独立编码器模块"""
    def __init__(self, in_channels, base_channels, num_downsample, latent_channels):
        super().__init__()
        self.encoder = nn.ModuleList()
        current_channels = in_channels
        for i in range(num_downsample):
            next_channels = base_channels * (2 ** i)
            self.encoder.append(
                nn.Sequential(
                    Down(current_channels, next_channels),
                    ASPP(next_channels, next_channels)
                )
            )
            current_channels = next_channels

        # 特征通道计算
        feature_channels = [base_channels * (2 ** i) for i in range(num_downsample - 3, num_downsample)]

        self.triple_asff = TripleASFF(feature_channels=feature_channels)
        self.latent_proj_mu = nn.Conv2d(feature_channels[-1], latent_channels, 3, padding=1)
        self.latent_proj_logvar = nn.Conv2d(feature_channels[-1], latent_channels, 3, padding=1)

    def forward(self, x):
        features = []
        for i, enc in enumerate(self.encoder):
            x = enc(x)
            if i >= len(self.encoder) - 3:
                features.append(x)

        # 检查特征图顺序
        assert len(features) == 3, "需要三个特征层进行融合"
        fused = self.triple_asff(features)

        # 输出潜在空间的均值和对数方差
        mu = self.latent_proj_mu(fused)
        logvar = self.latent_proj_logvar(fused)

        # 重参数化技巧
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        return z, mu, logvar

class VAE_Decoder(nn.Module):
    """独立解码器模块（修复通道问题）"""
    def __init__(
        self,
        latent_channels=4,
        base_channels=32,
        num_upsample=4,
        out_channels=1  # 修改为与输入通道一致
    ):
        super().__init__()
        # 计算各层通道数（需与编码器对称）
        feature_channels = [base_channels*(2**i) for i in range(1, num_upsample+1)][::-1]
        
        # 上采样路径
        self.up_blocks = nn.ModuleList()
        current_channels = latent_channels
        for ch in feature_channels:
            self.up_blocks.append(UpBlock(current_channels, ch))
            current_channels = ch
        
        # 最终输出层
        self.final_conv = nn.Conv2d(feature_channels[-1], out_channels, 3, padding=1)

    def forward(self, x):
        for block in self.up_blocks:
            x = block(x)
        return self.final_conv(x)  # 输出维度与输入一致

class VAE_UNet_MultiASFF(nn.Module):
    """整合后的VAE（支持参数冻结）"""
    def __init__(
        self,
        in_channels=3,
        latent_channels=4,
        base_channels=32,
        num_downsample=4,
        scale_factor=0.18215,
    ):
        super().__init__()
        self.encoder = VAE_Encoder(in_channels, base_channels, num_downsample, latent_channels)
        self.decoder = VAE_Decoder(latent_channels, base_channels, num_downsample, 1)
        self.scale_factor = scale_factor
        


    def forward(self, x):
        z, mu, logvar = self.encoder(x)
        return z,mu, logvar, self.decoder(z)

class UpBlock(nn.Module):
    """上采样块（增加通道控制）"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.up(x)

def load_pretrained_vae_unet(model_path):
    model = VAE_UNet_MultiASFF(
        in_channels=2,
        latent_channels=4,
        base_channels=32,
        num_downsample=3,
    )
    checkpoint = torch.load(model_path)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

def encode_with_pretrained_vae_unet(model, rgb_in):
    with torch.no_grad():
        z, mu, logvar = model.encoder(rgb_in)
        rgb_latent= z * model.scale_factor
    return  rgb_latent

def decode_with_pretrained_vae_unet(model, latent):
    with torch.no_grad():
        latent = latent / model.scale_factor
        recon = model.decoder(latent)
    return recon


if __name__ == "__main__":
    # 验证配置
    # model = VAE_UNet_MultiASFF(
    #     in_channels=2,
    #     latent_channels=4,
    #     base_channels=32,
    #     num_downsample=3,
    # )
    
    # dummy_input = torch.randn(2, 3, 128, 128)
    # latent,mu, logvar, recon = model(dummy_input)
    # print(f"潜在空间尺寸: {latent.shape}")  # 应输出 torch.Size([2, 4, 8, 8])
    # print(f"重建结果尺寸: {recon.shape}")   # 应输出 torch.Size([2, 1, 128, 128])
    model_path = "/home/yinjun/project/Marigold-main/vae_mult/VAE_UNet_MultiASFF.pth"
    model = load_pretrained_vae_unet(model_path)
    
    # 测试编码器
    dummy_input = torch.randn(2, 2, 128, 128)
    z, mu, logvar = encode_with_pretrained_vae_unet(model, dummy_input)
    print(f"潜在空间尺寸: {z.shape}")  # 应输出 torch.Size([2, 4, 8, 8])
    
    # 测试解码器
    recon = decode_with_pretrained_vae_unet(model, z)
    print(f"重建结果尺寸: {recon.shape}")   # 应输出 torch.Size([2, 1, 128, 128])