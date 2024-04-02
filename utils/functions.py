import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import gaussian_filter




def cal_loss(fs_list, ft_list):
    t_loss = 0
    N = len(fs_list)
    for i in range(N):
        fs = fs_list[i]
        ft = ft_list[i]
        _, _, h, w = fs.shape
        fs_norm = F.normalize(fs, p=2)
        ft_norm = F.normalize(ft, p=2)
        f_loss = 0.5 * (ft_norm - fs_norm) ** 2
        f_loss = f_loss.sum() / (h * w)
        t_loss += f_loss

    return t_loss / N


def cal_anomaly_maps(fs_list, ft_list, out_size):
    anomaly_map = 0
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        fs_norm = F.normalize(fs, p=2)
        ft_norm = F.normalize(ft, p=2)
        _, _, h, w = fs.shape

        a_map = (0.5 * (ft_norm - fs_norm) ** 2) / (h * w)

        a_map = a_map.sum(1, keepdim=True)

        a_map = F.interpolate(
            a_map, size=out_size, mode="bilinear", align_corners=False
        )
        anomaly_map += a_map
    anomaly_map = anomaly_map.squeeze().cpu().numpy()
    for i in range(anomaly_map.shape[0]):
        anomaly_map[i] = gaussian_filter(anomaly_map[i], sigma=4)

    return anomaly_map


def cal_anomaly_maps_RnetEffNet(fs_res, ft_res,fs_eff,ft_eff, out_size):
    anomaly_map = 0
    anomaly_map_effNet=0
    for i in range(len(ft_res)):
        fs = fs_res[i]
        ft = ft_res[i]
        fs_norm = F.normalize(fs, p=2)
        ft_norm = F.normalize(ft, p=2)
        _, _, h, w = fs.shape

        a_map = (0.5 * (ft_norm - fs_norm)**2) / (h*w)

        a_map = a_map.sum(1, keepdim=True)

        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=False)



        anomaly_map += a_map
    
    # EffNet part
    for i in range(len(ft_eff)):
        fs = fs_eff[i]
        ft = ft_eff[i]
        fs_norm = F.normalize(fs, p=2)
        ft_norm = F.normalize(ft, p=2)
        _, _, h, w = fs.shape
        a_map_effNet = (0.5 * (ft_norm - fs_norm)**2) / (h*w)

        a_map_effNet = a_map_effNet.sum(1, keepdim=True)

        a_map_effNet = F.interpolate(a_map_effNet, size=out_size, mode='bilinear', align_corners=False)
        anomaly_map_effNet += a_map_effNet
    stat_scope_factor=torch.max(anomaly_map_effNet)-torch.min(anomaly_map_effNet)
    anomaly_map_effNet = F.normalize(anomaly_map_effNet, dim=2)

    anomaly_map=anomaly_map*(stat_scope_factor*anomaly_map_effNet)

    anomaly_map = anomaly_map.squeeze().cpu().numpy()
    for i in range(anomaly_map.shape[0]):
        anomaly_map[i] = gaussian_filter(anomaly_map[i], sigma=4)

    return anomaly_map
