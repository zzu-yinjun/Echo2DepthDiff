import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2Model, ViTModel
from torchvision import transforms
import librosa
import numpy as np
from PIL import Image
 
from vit_pytorch import ViT 
# -------------------
# 1. 回声编码器（基于您提供的 CIDE 类）
# -------------------
class CIDE(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        # 输入处理: [B, 2, L] -> [B, 64, L/4]
        self.input_conv = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
        )
        
        # 残差块
        self.res_blocks = nn.ModuleList([
            ResBlock1D(64, 64) for _ in range(2)
        ])
        
        # 下采样块
        self.down_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(64 * (2**i), 64 * (2**(i+1)), kernel_size=3, stride=2, padding=1),
                nn.BatchNorm1d(64 * (2**(i+1))),
                nn.GELU()
            ) for i in range(3)  # 64->128->256->512
        ])
        
        # 使用自注意力机制增强特征提取
        self.attn = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
        
        # 输出处理
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(512, 400),
            nn.BatchNorm1d(400),
            nn.GELU(),
            nn.Linear(400, 100),
            nn.BatchNorm1d(100)
        )
        
        self.dim = emb_dim
        self.m = nn.Softmax(dim=1)
        self.embeddings = nn.Parameter(torch.randn(100, self.dim))
        # self.embedding_adapter = EmbeddingAdapter(emb_dim=self.dim)
        self.embedding_adapter = nn.Sequential(
            nn.Linear(self.dim, self.dim*2),
            nn.LayerNorm(self.dim*2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(self.dim*2, self.dim),
            nn.LayerNorm(self.dim)  # 添加输出归一化
        )
        self.gamma = nn.Parameter(torch.ones(self.dim) * 1e-4)
        # self.additional_layers = nn.Sequential(
        #     nn.Linear(768, 1024),
        #     nn.LayerNorm(1024),
        #     nn.GELU(),
        #     nn.Dropout(0.2),    # 增加dropout率
        #     nn.Linear(1024, 1024),
        #     nn.LayerNorm(1024),
        #     nn.GELU(),
        #     nn.Dropout(0.2),
        #     nn.Linear(1024, 768)
        # )
            
    def forward(self, x):
        # 初始卷积处理
        x = self.input_conv(x)  # [B, 2, L] -> [B, 64, L/4]
        
        # 残差块处理
        for res_block in self.res_blocks:
            x = res_block(x)
        
        # 下采样处理
        for down_block in self.down_blocks:
            x = down_block(x)  # 最终 [B, 512, L/32]
        
        # 自注意力机制增强特征
        x_trans = x.permute(0, 2, 1)  # [B, L/32, 512]
        attn_out, _ = self.attn(x_trans, x_trans, x_trans)
        x = attn_out.permute(0, 2, 1) + x  # 残差连接
        
        # 全局池化
        x = self.global_pool(x)  # [B, 512, 1]
        x = x.squeeze(-1)  # [B, 512]
        
        # 生成类别概率和嵌入
        class_probs = self.fc(x)
        class_probs = self.m(class_probs)
        class_embeddings = class_probs @ self.embeddings
        
        # 使用新的嵌入适配器和归一化
        conditioning_embedding = self.embedding_adapter(class_embeddings)
        
        # 显式L2归一化，确保嵌入向量在单位超球面上
        conditioning_embedding = F.normalize(conditioning_embedding, p=2, dim=-1)
        
        return conditioning_embedding

# 残差块定义
class ResBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.activation = nn.GELU()
        
        # 如果输入输出通道数不同，添加一个1x1卷积进行调整
        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        out = self.activation(out)
        return out
    

class DepthViTEncoder(nn.Module):
    def __init__(self, emb_dim=768):
        super().__init__()
        # 使用vit_pytorch库创建ViT
        self.vit = ViT(
            image_size=128,
            patch_size=8,
            num_classes=emb_dim,  # 直接输出emb_dim维度的特征
            dim=emb_dim,
            depth=6, 
            heads=12,
            mlp_dim=1024,
            channels=1,
            dropout=0.1,
            pool='cls'  # 使用cls token进行池化
        )
        # 简单的投影层
        self.depth_proj = nn.Sequential(
            nn.Linear(emb_dim, emb_dim*2),
            nn.LayerNorm(emb_dim*2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(emb_dim*2, emb_dim),
            nn.LayerNorm(emb_dim)
        )

    def forward(self, x):
        # ViT处理深度图
        x = self.vit(x)
        # 投影
        x = self.depth_proj(x)
        # 归一化
        x = F.normalize(x, p=2, dim=-1)
        return x
        
        return features
# -------------------
# 3. 对比学习框架
# -------------------
class EchoDepthContrastiveModel(nn.Module):
    def __init__(self, echo_emb_dim=768, depth_emb_dim=768):
        super().__init__()
        self.echo_encoder = CIDE(emb_dim=echo_emb_dim)
        self.depth_encoder = DepthViTEncoder(emb_dim=depth_emb_dim)
        # 使用较低的温度参数
        self.temperature = nn.Parameter(torch.tensor(0.07))
    
    def forward(self, echo, depth):
        # 获取两种模态的嵌入向量
        echo_embed = self.echo_encoder(echo)
        depth_embed = self.depth_encoder(depth)
        return echo_embed, depth_embed
    
    def contrastive_loss(self, echo_embeds, depth_embeds):
        # 处理可能的维度问题
        if echo_embeds.dim() > 2:
            echo_embeds = echo_embeds.squeeze(1)
        if depth_embeds.dim() > 2:
            depth_embeds = depth_embeds.squeeze(1)
        
        # 归一化嵌入向量
        echo_embeds = F.normalize(echo_embeds, dim=1)
        depth_embeds = F.normalize(depth_embeds, dim=1)
        
        batch_size = echo_embeds.shape[0]
        
        # 计算相似度矩阵
        sim_matrix = torch.matmul(echo_embeds, depth_embeds.T) / self.temperature
        
        # 对比学习标签 - 对角线位置为匹配对
        labels = torch.arange(batch_size, device=sim_matrix.device)
        
        # 计算匹配对的相似度（对角线元素）
        pos_sim = torch.diag(sim_matrix)
        
        # 双向对比损失
        loss_e2d = F.cross_entropy(sim_matrix, labels)
        loss_d2e = F.cross_entropy(sim_matrix.T, labels)
        
        # 直接相似度损失 - 强制匹配对相似度接近1
        # 注意：这里使用原始的归一化嵌入计算相似度，而非相似度矩阵的对角线
        # 因为相似度矩阵已经除以了温度参数
        pos_sim_raw = torch.sum(echo_embeds * depth_embeds, dim=1)
        direct_sim_loss = torch.mean(1.0 - pos_sim_raw)
        
        # 组合损失 - 使用高权重的直接相似度损失
        alpha = 1.0  # 高权重直接相似度损失
        total_loss = loss_e2d + loss_d2e + alpha * direct_sim_loss
        
        return total_loss
  
 

