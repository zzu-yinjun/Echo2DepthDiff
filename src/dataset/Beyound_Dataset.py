import os.path
import re
import time
import librosa
import random
import math
import numpy as np
import glob
import torch
import pickle
from PIL import Image, ImageEnhance
import torchvision.transforms as transforms
import torch.utils.data as data
from scipy.io.wavfile import read as read_wav
from scipy.signal import stft


def get_scene_list(path):
    with open(path) as f:
        scenes_test = f.readlines()
    scenes_test = [x.strip() for x in scenes_test]
    return scenes_test


def normalize(samples, desired_rms=0.1, eps=1e-4):
    rms = np.maximum(eps, np.sqrt(np.mean(samples ** 2)))
    samples = samples * (desired_rms / rms)
    return samples


def generate_spectrogram(audioL, audioR, winl=32):
    channel_1_spec = librosa.stft(audioL, n_fft=512, win_length=winl)
    channel_2_spec = librosa.stft(audioR, n_fft=512, win_length=winl)

    spectro_two_channel = np.concatenate(
        (np.expand_dims(np.abs(channel_1_spec), axis=0), np.expand_dims(np.abs(channel_2_spec), axis=0)), axis=0)
    # print(spectro_two_channel.shape)
    return spectro_two_channel


def get_file_list(path):
    file_name_prefix = []
    for file in os.listdir(path):
        file_name_prefix.append(file.split('.')[0])
    file_name_prefix = np.unique(file_name_prefix)
    return file_name_prefix


def add_to_list(index_list, file_path, data):
    for index in index_list:
        rgb = os.path.join(file_path, index) + '.png'
        audio = os.path.join(file_path, index) + '.wav'
        depth = os.path.join(file_path, index) + '.npy'
        data.append([rgb, audio, depth])


class AudioVisualDataset(data.Dataset):
    def __init__(self, dataset, mode):
        super(AudioVisualDataset, self).__init__()
        self.train_data = []
        self.val_data = []
        self.test_data = []
        replica_dataset_path ="/home/yinjun/dataset/replica-dataset"
        mp3d_dataset_path = "/data/yinjun/mp3d-dataset"
        metadata_path = "/data/yinjun/metadata/mp3d"
        self.audio_resize = True
        if dataset == 'mp3d':
            self.win_length = 32
            self.audio_sampling_rate = 16000
            self.audio_length = 0.060
            # train,val,test scenes
            train_scenes_file = os.path.join(metadata_path, 'mp3d_scenes_train.txt')
            val_scenes_file = os.path.join(metadata_path, 'mp3d_scenes_val.txt')
            test_scenes_file = os.path.join(metadata_path, 'mp3d_scenes_test.txt')
            train_scenes = get_scene_list(train_scenes_file)
            val_scenes = get_scene_list(val_scenes_file)
            test_scenes = get_scene_list(test_scenes_file)
            for scene in os.listdir(mp3d_dataset_path):
                if scene in train_scenes:
                    for orn in os.listdir(os.path.join(mp3d_dataset_path, scene)):
                        file_name_prefix = get_file_list(os.path.join(mp3d_dataset_path, scene, orn))
                        add_to_list(file_name_prefix, os.path.join(mp3d_dataset_path, scene, orn), self.train_data)
                elif scene in val_scenes:
                    for orn in os.listdir(os.path.join(mp3d_dataset_path, scene)):
                        file_name_prefix = get_file_list(os.path.join(mp3d_dataset_path, scene, orn))
                        add_to_list(file_name_prefix, os.path.join(mp3d_dataset_path, scene, orn), self.val_data)
                elif scene in test_scenes:
                    for orn in os.listdir(os.path.join(mp3d_dataset_path, scene)):
                        file_name_prefix = get_file_list(os.path.join(mp3d_dataset_path, scene, orn))
                        add_to_list(file_name_prefix, os.path.join(mp3d_dataset_path, scene, orn), self.test_data)
        if dataset == 'replica':
            self.win_length = 64
            self.audio_sampling_rate = 44100
            self.audio_length = 0.060
            # apartment 2, frl apartment 5, and office 4 are test scenes
            for scene in os.listdir(replica_dataset_path):
                if scene not in ['apartment_2', 'frl_apartment_5', 'office_4']:
                    # 训练集
                    for orn in os.listdir(os.path.join(replica_dataset_path, scene)):
                        file_name_prefix = get_file_list(os.path.join(replica_dataset_path, scene, orn))
                        add_to_list(file_name_prefix, os.path.join(replica_dataset_path, scene, orn), self.train_data)
                else:
                    for orn in os.listdir(os.path.join(replica_dataset_path, scene)):
                        file_name_prefix = get_file_list(os.path.join(replica_dataset_path, scene, orn))
                        val = file_name_prefix[:len(file_name_prefix) // 2]
                        test = file_name_prefix 
                        # test = file_name_prefix[len(file_name_prefix)]
                        add_to_list(val, os.path.join(replica_dataset_path, scene, orn), self.val_data)
                        add_to_list(test, os.path.join(replica_dataset_path, scene, orn), self.test_data)

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        vision_transform_list = [transforms.ToTensor(), normalize]
        self.vision_transform = transforms.Compose(vision_transform_list)
        self.mode = mode
        self.dataset = dataset

    def __getitem__(self, index):
        # rgb, audio, depth
        if self.mode == 'train':
            data_ = self.train_data[index]
        elif self.mode == 'val':
            data_ = self.val_data[index]
        elif self.mode == 'test':
            data_ = self.test_data[index]
        # print(data_)
        rgb_path, audi_path, depth_path = data_[0], data_[1], data_[2]

        # 读取RGB图像并转换为NumPy数组
        rgb = Image.open(rgb_path).convert('RGB')

 
        # 将PIL图像转换为numpy数组
        rgb_np = np.array(rgb).astype(np.float32)
        rgb_original = rgb_np.copy()
        # 归一化到[-1, 1]范围
        rgb_norm = rgb_np / 255.0 * 2.0 - 1.0  #  [0, 255] -> [-1, 1]
        # 转换为PyTorch张量，并调整通道顺序为[C, H, W]
        rgb_norm = np.transpose(rgb_norm, (2, 0, 1))
        rgb_norm = torch.from_numpy(rgb_norm).float()

        # 处理音频数据
        audio, audio_rate = librosa.load(audi_path, sr=self.audio_sampling_rate, mono=False, duration=self.audio_length)
        audio = normalize(audio)
        audio_spec_both = torch.FloatTensor(generate_spectrogram(audio[0, :], audio[1, :], self.win_length))
        audio_min = audio_spec_both.min()
        audio_max = audio_spec_both.max()
        if audio_max > audio_min:  # 避免除以零
            audio_spec_both = 2.0 * (audio_spec_both - audio_min) / (audio_max - audio_min) - 1.0

        if self.audio_resize:
            audio_spec_both = transforms.Resize((128, 128))(audio_spec_both)
        audio_spec_both = torch.cat([audio_spec_both, audio_spec_both[:1]], dim=0) 
        
        
        # 处理深度图
        depth = torch.FloatTensor(np.load(depth_path))

        # 添加通道维度
        depth = depth.unsqueeze(0)

        # 创建有效深度掩码（排除小于0的部分）
        valid_mask = depth > 0

        depth_original = depth.clone()
        

        # 如果需要归一化深度图，可以参考如下代码
        # 方法1: 将深度值归一化到[-1, 1]范围
        min_depth = 0.1  # 最小有效深度
        max_depth = 14  # 最大有效深度（可根据你的数据集调整）

        # 只对有效范围内的深度值进行归一化
        depth_normalized = depth.clone()

        # 将掩码区域内的值归一化到[-1, 1]
        depth_normalized[valid_mask] = 2.0 * (depth[valid_mask] - min_depth) / (max_depth - min_depth) - 1.0

        # 将无效区域设置为特定值（例如-1，表示远处）
        depth_normalized[~valid_mask] = -1.0

        return {
        'img': rgb_norm, 
        'audio_spec': audio_spec_both, 
        'audio_wave': audio,
        'depth': depth_normalized, 
        'valid_mask_raw': valid_mask,
        'rgb_original': rgb_original,
        'depth_original': depth_original,
        'rgb_path': rgb_path  # 可选：添加原始路径以便直接读取
    }
    def __len__(self):
        if self.mode == 'train':
            return len(self.train_data)
        elif self.mode == 'val':
            return len(self.val_data)
        elif self.mode == 'test':
            return len(self.test_data)


def get_data_loader(dataset_name, mode, shuffle, config):
    dataset = AudioVisualDataset(dataset_name, mode, config)
    return data.DataLoader(dataset, batch_size=config.batch_size, shuffle=shuffle, num_workers=config.num_workers)


class Config(object):
    def __init__(self):
        self.expr_dir = 'model_mp3d'
        self.lr_visual = 0.0001
        self.lr_audio = 0.0001
        self.lr_attention = 0.0001
        self.lr_material = 0.0001
        self.learning_rate_decrease_itr = -1
        self.decay_factor = 0.94
        self.optimizer = 'adam'
        self.weight_decay = 0.0001
        self.beta1 = 0.9
        self.batch_size = 100
        self.epochs = 50
        self.dataset = 'mp3d'
        self.checkpoints_dir = ''
        self.device = 'cuda'
        self.num_workers = 4
        self.init_material_weight = '/home/malong/Reproduction-Beyond/model_pth/material_pre_trained_minc.pth'
        self.replica_dataset_path = '/home/malong/Reproduction-Beyond/dataset/replica-dataset'
        self.mp3d_dataset_path = '/home/malong/Reproduction-Beyond/dataset/mp3d-dataset'
        self.metadata_path = '/home/malong/Reproduction-Beyond/dataset/metadata/mp3d'
        if self.dataset == 'replica':
            self.max_depth = 14.104
            self.audio_shape = [2, 257, 166]
        else:
            self.max_depth = 10.0
            self.audio_shape = [2, 257, 121]
        self.modo = 'train'
        self.display_freq = 1
        self.validation_freq = 1

config = Config()
config.dataset = 'replica'
config.modo = 'test'

a = AudioVisualDataset(config.dataset, config.modo)
d = a[1]
# print(f"Original depth max: {d['depth_original'].max()}, min: {d['depth_original'].min()}")
print(f"Normalized depth max: {d['depth'].max()}, min: {d['depth'].min()}")
# depth_max_values = []

# for i in range(len(a)):
#     data = a[i]
#     depth_max_values.append(data['depth_original'].max().item())
# # 4.111029589009252
# average_max_depth = sum(depth_max_values) / len(depth_max_values)
# print(f"Average max depth value: {average_max_depth}")