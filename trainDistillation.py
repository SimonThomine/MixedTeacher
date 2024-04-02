import os
import argparse
import time
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from datasets.mvtec import MVTecDataset
from utils.util import AverageMeter
from utils.functions import (
    cal_loss,
    cal_anomaly_maps,
)
from models.resnet_reduced_backbone import reduce_student18
from models.studentEffNet import studentEffNet
from models.teacherTimm import teacherTimm

class AnomalyDistillation:
    def __init__(self, args=None,modelName="reducedResnet18"): #efficientnet_b0-custom reducedResnet18
        if args != None:
            self.device = args.device
            self.data_path = args.data_path
            self.obj = args.obj
            self.img_resize = args.img_resize
            self.img_cropsize = args.img_cropsize
            self.validation_ratio = args.validation_ratio
            self.num_epochs = args.num_epochs
            self.lr = args.lr
            self.batch_size = args.batch_size
            self.vis = args.vis
            self.model_dir = args.model_dir
            self.img_dir = args.img_dir
            self.modelName = modelName

            self.load_model()
            self.load_dataset()

            self.optimizer = torch.optim.Adam(self.model_s.parameters(), lr=self.lr, betas=(0.9, 0.999))
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer,max_lr=self.lr*10,epochs=self.num_epochs,steps_per_epoch=len(self.train_loader),)

    def load_dataset(self):
        kwargs = (
            {"num_workers": 4, "pin_memory": True} if torch.cuda.is_available() else {}
        )
        train_dataset = MVTecDataset(
            self.data_path,
            class_name=self.obj,
            is_train=True,
            resize=self.img_resize,
            cropsize=self.img_cropsize,
        )
        img_nums = len(train_dataset)
        valid_num = int(img_nums * self.validation_ratio)
        train_num = img_nums - valid_num
        train_data, val_data = torch.utils.data.random_split(
            train_dataset, [train_num, valid_num]
        )
        self.train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=self.batch_size, shuffle=True, **kwargs
        )
        self.val_loader = torch.utils.data.DataLoader(
            val_data, batch_size=8, shuffle=False, **kwargs
        )

    def load_model(self):
        print("loading and training " + self.modelName)
        if self.modelName == "efficientnet_b0":
            self.model_t= teacherTimm(backbone_name="efficientnet_b0",out_indices=[3,4]).to(self.device)
            self.model_s = studentEffNet().to(self.device)
            
        elif self.modelName == "reducedResnet18":
            self.model_t = teacherTimm(backbone_name="resnet18",out_indices=[1,2]).to(self.device)
            self.model_s = reduce_student18(pretrained=False).to(self.device)
        
        for param in self.model_t.parameters():
            param.requires_grad = False
        self.model_t.eval()

    def train(self):
        print("training " + self.obj)
        self.model_s.train()
        best_score = None
        start_time = time.time()
        epoch_time = AverageMeter()
        epoch_bar = tqdm(total=len(self.train_loader) * self.num_epochs,desc="Training",unit="batch")
        for epoch in range(1, self.num_epochs + 1):
            losses = AverageMeter()
            for data, label, _ in self.train_loader:
                data = data.to(self.device)

                self.optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    features_t, features_s = self.infer(data)
                    loss = cal_loss(features_s, features_t)
                    losses.update(loss.sum().item(), data.size(0))
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()
                epoch_bar.set_postfix({"Loss": loss.item()})
                epoch_bar.update()

            val_loss = self.val(epoch, epoch_bar)
            if best_score is None:
                best_score = val_loss
                self.save_checkpoint()
            elif val_loss < best_score:
                best_score = val_loss
                self.save_checkpoint()

            epoch_time.update(time.time() - start_time)
            start_time = time.time()
        epoch_bar.close()
        print("Training end.")

    def val(self, epoch, epoch_bar):
        self.model_s.eval()
        losses = AverageMeter()
        for data, _, _ in self.val_loader:
            data = data.to(self.device)
            with torch.set_grad_enabled(False):
                features_t, features_s = self.infer(data)
                loss = cal_loss(features_s, features_t)
                losses.update(loss.item(), data.size(0))
        epoch_bar.set_postfix({"Loss": loss.item()})

        return losses.avg

    def save_checkpoint(self):
        state = {"model": self.model_s.state_dict()}
        torch.save(state, os.path.join(self.model_dir, "model_s.pth"))

    def test(self):
        try:
            checkpoint = torch.load(os.path.join(self.model_dir, "model_s.pth"))
        except:
            raise Exception("Check saved model path.")
        self.model_s.load_state_dict(checkpoint["model"])
        self.model_s.eval()

        kwargs = (
            {"num_workers": 4, "pin_memory": True} if torch.cuda.is_available() else {}
        )
        test_dataset = MVTecDataset(
            self.data_path,
            class_name=self.obj,
            is_train=False,
            resize=self.img_resize,
            cropsize=self.img_cropsize,
        )
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
                timeBefore = time.perf_counter()
                features_t, features_s = self.infer(data)
                timeAfterFeatures = time.perf_counter()
                print("inference : " + str(timeAfterFeatures - timeBefore))

                score = cal_anomaly_maps(features_s,features_t,self.img_cropsize)
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

    def infer(self, data):
        features_t = self.model_t(data)
        features_s = self.model_s(data)
        return features_t, features_s


def get_args():
    parser = argparse.ArgumentParser(description="Distillation")
    parser.add_argument("--phase", default="train")
    parser.add_argument(
        "--data_path", type=str, default="../../datasets/MVTECBase"
    )
    parser.add_argument("--obj", type=str, default="wood")
    parser.add_argument("--img_resize", type=int, default=256)  
    parser.add_argument("--img_cropsize", type=int, default=256)
    parser.add_argument("--validation_ratio", type=float, default=0.2)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.0004)  
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--vis", type=eval, choices=[True, False], default=True)
    parser.add_argument("--save_path", type=str, default="./mvtec_results/KD_eff")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    args.model_dir = args.save_path + "/models" + "/" + args.obj
    args.img_dir = args.save_path + "/imgs" + "/" + args.obj
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.img_dir, exist_ok=True)

    anoDistill = AnomalyDistillation(args)
    if args.phase == "train":
        anoDistill.train()
        anoDistill.test()
    elif args.phase == "test":
        anoDistill.test()
    else:
        print("Phase argument must be train or test.")
