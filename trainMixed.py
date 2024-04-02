import os
import argparse
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from datasets.mvtec import MVTecDataset
from utils.functions import (
    cal_anomaly_maps,
    cal_anomaly_maps_RnetEffNet
)

from models.studentEffNet import studentEffNet
from models.resnet_reduced_backbone import reduce_student18
from trainDistillation import AnomalyDistillation
from models.teacherTimm import teacherTimm

class MixedTeacher:
    def __init__(self, args=None):
        if args != None:
            self.device = args.device
            self.data_path = args.data_path
            self.obj = args.obj
            self.img_resize = args.img_resize
            self.img_cropsize = args.img_cropsize
            self.validation_ratio = args.validation_ratio
            self.vis = args.vis
            self.model_dir = args.model_dir
            self.img_dir = args.img_dir
            self.modelName = "efficientnet-b0" 
            self.KD_effnetPath = ""
            self.KD_resnetPath=""

    def load_model(self):
        print("object : " +self.obj)
        if self.modelName == "efficientnet-b0":
            self.model_t_effNet = teacherTimm(backbone_name="efficientnet_b0",out_indices=[3,4]).to(self.device)
            self.model_t_resNet = teacherTimm(backbone_name="resnet18",out_indices=[1,2]).to(self.device)
            
            self.model_KD_effNet = studentEffNet().to(self.device)
            self.model_KD_resNet = reduce_student18(pretrained=False).to(self.device)
            
        for param in self.model_t_effNet.parameters():
            param.requires_grad = False
        self.model_t_effNet.eval()
        for param in self.model_t_resNet.parameters():
            param.requires_grad = False
        self.model_t_resNet.eval()
        try:
            checkpoint = torch.load(os.path.join(self.KD_effnetPath, "model_s.pth"))
        except:
            raise Exception("Check saved model path.")
        self.model_KD_effNet.load_state_dict(checkpoint["model"])
        for param in self.model_KD_effNet.parameters():
            param.requires_grad = False
        self.model_KD_effNet.eval()
        
        try:
            checkpoint = torch.load(os.path.join(self.KD_resnetPath, "model_s.pth"))
        except:
            raise Exception("Check saved model path.")
        self.model_KD_resNet.load_state_dict(checkpoint["model"])
        for param in self.model_KD_resNet.parameters():
            param.requires_grad = False
        self.model_KD_resNet.eval()
        
        

    def test(self):
        self.load_model()
        kwargs = (
            {"num_workers": 4, "pin_memory": True} if torch.cuda.is_available() else {}
        )
        test_dataset = MVTecDataset(self.data_path,class_name=self.obj,is_train=False,resize=self.img_resize,cropsize=self.img_cropsize)
        batch_size_test = 1 
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size_test, shuffle=False, **kwargs
        )
        scores = []
        test_imgs = []
        gt_list = []
        gt_mask_list = []
        progressBar = tqdm(test_loader)
        for data, label, mask in test_loader:
            test_imgs.extend(data.cpu().numpy())
            gt_list.extend(label.cpu().numpy())
            gt_mask_list.append(mask.squeeze().cpu().numpy())
            data = data.to(self.device)
            with torch.set_grad_enabled(False):
                
                #! KD resNet
                features_t_res = self.model_t_resNet(data)
                features_s_res = self.model_KD_resNet(data)
                
                #! KD effNet
                features_t_eff = self.model_t_effNet(data)
                features_s_eff = self.model_KD_effNet(data)
                
                score=cal_anomaly_maps_RnetEffNet(features_s_res,features_t_res,features_s_eff,features_t_eff,self.img_cropsize)   


                progressBar.update()

            if batch_size_test == 1:
                scores.append(score)
            else:
                scores.extend(score)
        progressBar.close()
        scores = np.asarray(scores)

        max_anomaly_score = scores.max()
        min_anomaly_score = scores.min()
        scores = (scores - min_anomaly_score) / (max_anomaly_score - min_anomaly_score)

        img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
        gt_list = np.asarray(gt_list)
        img_roc_auc = roc_auc_score(gt_list, img_scores)
        print(self.obj + " image ROCAUC: %.3f" % (img_roc_auc))

       


def get_args():
    parser = argparse.ArgumentParser(description="EfficientNet distillation")
    parser.add_argument("--phase", default="train")
    parser.add_argument(
        "--data_path", type=str, default="../../datasets/MVTEC"
    )
    parser.add_argument("--obj", type=str, default="wood")
    parser.add_argument("--img_resize", type=int, default=256)  
    parser.add_argument("--img_cropsize", type=int, default=256)  
    parser.add_argument("--validation_ratio", type=float, default=0.2)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.0004) 
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--vis", type=eval, choices=[True, False], default=True)
    parser.add_argument("--save_path", type=str, default="./mvtec_results")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    argsKd = args
    argsKd.model_dir = args.save_path + "/KD_eff/models" + "/" + args.obj
    argsKd.img_dir = args.save_path + "/KD_eff/imgs" + "/" + args.obj
    os.makedirs(argsKd.model_dir, exist_ok=True)
    os.makedirs(argsKd.img_dir, exist_ok=True)
    Kd_eff = AnomalyDistillation(argsKd,modelName="efficientnet_b0")

    argsKd = args
    argsKd.model_dir = args.save_path + "/KD_ReducedResnet/models" + "/" + args.obj
    argsKd.img_dir = args.save_path + "/KD_ReducedResnet/imgs" + "/" + args.obj
    os.makedirs(argsKd.model_dir, exist_ok=True)
    os.makedirs(argsKd.img_dir, exist_ok=True)
    Kd_res = AnomalyDistillation(argsKd,modelName="reducedResnet18")
    


    args.img_dir = args.save_path + "/MixedTeacher/imgs" + "/" + args.obj
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.img_dir, exist_ok=True)

    Combined = MixedTeacher(args)
    Combined.KD_effnetPath = Kd_eff.model_dir
    Combined.KD_resnetPath = Kd_res.model_dir
    if args.phase == "train":
        Kd_eff.train()
        Kd_res.train()
        Combined.test()
    elif args.phase == "test":
        Combined.test()
    else:
        print("Phase argument must be train or test.")


