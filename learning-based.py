import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import numpy as np
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import math

def rot_6d(x, y):
    # Gram-schmidt process
    x = F.normalize(x, dim=-1)
    y = y - x * (x * y).sum(-1, keepdims=True)
    y = F.normalize(y, dim=-1)
    z = torch.cross(x, y, -1)
    return torch.stack([x, y, z], dim=-1)


class PointNet(nn.Module):
    def __init__(self, global_feature_size=79):
        super(PointNet, self).__init__()
        self.conv1 = torch.nn.Conv1d(6, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)  
        self.bn3 = nn.BatchNorm1d(1024)

        self.branch = torch.nn.Linear(global_feature_size, 1024)

        self.dense1 = torch.nn.Linear(1024, 512)
        self.bn4 = nn.BatchNorm1d(512)
        self.dense2 = torch.nn.Linear(512, 256)
        self.bn5 = nn.BatchNorm1d(256)
        self.dense3 = torch.nn.Linear(256, 3 + 6) # translation + rotation

        for net in [
            self.conv1,
            self.conv2,
            self.conv3,
            self.dense1,
            self.dense2,
            self.dense3,
        ]:
            torch.nn.init.xavier_uniform_(net.weight)

    def forward(self, x, label):
        points = x[:, :3]  # batch_size, 3, n_points
        colors = x[:, 3:]

        # center(0,0,0) and rescale 归一化到[-1,1]
        mins = points.min(dim=2, keepdim=True).values
        maxs = points.max(dim=2, keepdim=True).values
        center = (mins + maxs) / 2
        half_extents = (maxs - mins) / 2
        longest = half_extents.max(dim=1, keepdim=True).values.clamp(
            min=1e-3
        )
        points = (points - center) / longest

        x = torch.cat([points, colors], dim=1)
        x = F.relu(self.bn1(self.conv1(x)))

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x)) 
        x = torch.max(x, 2, keepdim=True)[0] 
        x = x.view(-1, 1024)
        x = F.relu(self.bn4(self.dense1(x)))
        x = F.relu(self.bn5(self.dense2(x)))
        x = self.dense3(x)

        trans = x[..., :3]
        rot = rot_6d(x[..., 3:6], x[..., 6:9])

        # scale back and un-center (batch_size, 3)
        trans = trans * longest.view(-1, 1) + center.view_as(trans)

        return trans, rot

class MyDataset(Dataset):
    def __init__(self, root, n_points=256, debug=False):
        self.n_points = n_points
        self.root = Path(root)
        self.files = list(self.root.iterdir())
        if debug:
            self.files = self.files[:128]
        np.random.shuffle(self.files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        result = {}
        npz = np.load(self.files[index])

        object_id = npz["id"]
        points = npz["points"]
        colors = npz["colors"]
        pose = npz["pose"]

        if len(points) == 0:
            points = np.zeros((self.n_points, 3))
            colors = np.zeros((self.n_points, 3))
        if len(points) < self.n_points:
            idx = np.random.choice(len(points), self.n_points, replace=True)
            points = points[idx]
            colors = colors[idx]
        else:
            idx = np.random.choice(len(points), self.n_points, replace=False)
            points = points[idx]
            colors = colors[idx]
        result["object_id"] = torch.tensor(object_id, dtype=torch.long)
        result["points"] = torch.tensor(points.T, dtype=torch.float32)
        result["colors"] = torch.tensor(colors.T, dtype=torch.float32)
        result["pose"] = torch.tensor(pose, dtype=torch.float32)
        result["name"] = self.files[index].name.split("_")[0]

        return result

def rotation_matrix_gen(alpha=0.0, beta=0.0, gamma=0.0):
    """
    return rotation by x,y,z axis
    """
    alpha = torch.tensor(alpha)
    beta = torch.tensor(beta)
    gamma = torch.tensor(gamma)
    rotation_matrix =  \
            torch.tensor([beta.cos()*gamma.cos(), alpha.sin()*beta.sin()*gamma.cos()-alpha.cos()*gamma.sin(), alpha.cos()*beta.sin()*gamma.cos()+alpha.sin()*gamma.sin(),
                          beta.cos()*gamma.sin(), alpha.sin()*beta.sin()*gamma.sin()+alpha.cos()*gamma.cos(), alpha.cos()*beta.sin()*gamma.sin()-alpha.sin()*gamma.cos(),
                          -beta.sin(), alpha.sin()*beta.cos(), alpha.cos()*beta.cos()]).to(device)
    rotation_matrix = torch.reshape(rotation_matrix, (3, 3))
    return rotation_matrix  

def find_symmetry_pose(pose, object_id, inf_num = 20):
    """
    return all the symmetry Rotaton 3*3
    """
    symmetry_pose = torch.Tensor(pose.shape[0],3,3,21).to(device)
    for i in range(object_id.shape[0]):
        ob_id = object_id[i].item()
        num = 0
        symmetry_pose[i,:,:,num] = pose[i]
        if objects_data[ob_id]['geometric_symmetry'] == "no":
            num = num + 1
        else:
            symm_tokens = objects_data[ob_id]['geometric_symmetry'].split("|")
            for symm_property in symm_tokens:
                axis, order = symm_property[0], symm_property[1:]
                order_num = 0
                # 计算沿其中一个轴的对称数
                if(order == "inf"):
                    order_num = inf_num
                else:
                    order_num = int(order)
                # 计算沿某个轴对称后的rotation matrix
                if(axis == "x"):
                    for j in range(1, order_num):
                        symmetry_pose[i,:,:,num]=(rotation_matrix_gen(alpha=
                            float(j)*float(2*math.pi/order_num))@ pose[i])
                elif(axis == "y"):
                    for j in range(1, order_num):
                        symmetry_pose[i,:,:,num]=(rotation_matrix_gen(beta=
                            float(j)*float(2*math.pi/order_num))@ pose[i])
                elif(axis == "z"):
                    for j in range(1, order_num):
                        symmetry_pose[i,:,:,num] =(rotation_matrix_gen(gamma=
                            float(j)*float(2*math.pi/order_num))@ pose[i])
                num = num + 1
        while num < 21:
            symmetry_pose[i,:,:,num]=pose[i]
            num = num + 1
    return symmetry_pose

def cal_symm_loss_rot(pred_rot, gt_rot):
    """
    min of N loss
    """
    gt_rot = find_symmetry_pose(pose[:, :3, :3], object_id, inf_num=20)
    loss_rot = ((pred_rot.unsqueeze(dim=-1) - gt_rot).abs())\
                .sum(dim=(1,2))\
                .min(dim=1)[0]\
                .sum()/pose.numel() # pose的shape
    return loss_rot

batch_size = 1024
num_workers = 4
epochs = 35
init_lr = 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PointNet().to(device)
optim = torch.optim.Adam(model.parameters(), lr=init_lr)

train_data = MyDataset("../train")  # generated by make_data.py
valid_data = MyDataset("../val")

loader_train = DataLoader(train_data, batch_size=batch_size, num_workers=num_workers)
loader_val = DataLoader(valid_data, batch_size=batch_size, num_workers=num_workers)

train_epochs_loss_rot = []
train_epochs_loss_trans = []
val_epochs_loss_rot = []
val_epochs_loss_trans = []

for epoch in range(epochs):
    with tqdm(total=len(loader_train)) as t:
        t.set_description("Epoch %i"%epoch)
        # train
        model.train()
        train_loss_trans = []
        train_loss_rot = []
        for iteration, data in enumerate(loader_train):
            object_id = data["object_id"].to(device)
            points = data["points"].to(device)
            colors = data["colors"].to(device)
            pose = data["pose"].to(device)

            pred_trans, pred_rot = model(
                torch.cat([points, colors], dim=1), F.one_hot(object_id, 79).float()
            )

            gt_trans = pose[:, :3, 3]
            gt_rot = pose[:, :3, :3]

            loss_trans = F.l1_loss(pred_trans, gt_trans)
            loss_rot = cal_symm_loss_rot(pred_rot, gt_rot)
            train_loss_trans.append(loss_trans.item())
            train_loss_rot.append(loss_rot.item())
            loss = loss_trans + loss_rot

            optim.zero_grad()
            loss.backward()
            optim.step()
            t.set_postfix(loss_trans=np.mean(train_loss_trans), loss_rot=np.mean(train_loss_rot))
            t.update(1)
            
        train_epochs_loss_rot.append(np.mean(train_loss_rot))
        train_epochs_loss_trans.append(np.mean(train_loss_trans))
        # valid
        model.eval()
        val_loss_rot = []
        val_loss_trans = []
        for data in loader_val:
            object_id = data["object_id"].to(device)
            points = data["points"].to(device)
            colors = data["colors"].to(device)
            pose = data["pose"].to(device)
            pred_trans, pred_rot = model(
                torch.cat([points, colors], dim=1), F.one_hot(object_id, 79).float()
            )
            gt_trans = pose[:, :3, 3]
            gt_rot = pose[:, :3, :3]

            loss_trans = F.l1_loss(pred_trans, gt_trans)
            loss_rot = cal_symm_loss_rot(pred_rot, gt_rot)
            val_loss_trans.append(loss_trans.item())
            val_loss_rot.append(loss_rot.item())
            loss = loss_trans + loss_rot
            
        if len(val_epochs_loss_rot)>=1 and np.mean(val_loss_rot) < val_epochs_loss_rot[-1]:
            torch.save(model, f"../model_save/model_{epoch}.pth")
        val_epochs_loss_rot.append(np.mean(val_loss_rot))
        val_epochs_loss_trans.append(np.mean(val_loss_trans))
    print("val_loss_rot="+str(np.mean(val_loss_rot))+" val_loss_trans="+str(np.mean(val_loss_trans)))
    if epoch == epochs - 1:
        torch.save(model, f"../model_save/model_{epoch}.pth")

# draw loss
plt.figure(figsize=(12,4))
plt.subplot(121)
plt.plot(train_epochs_loss_rot[0:],'-o',label="train_loss_rot")
plt.plot(val_epochs_loss_rot[0:],'-o',label="valid_loss_rot")
plt.title("epochs_loss_rot")
plt.legend()
plt.subplot(122)
plt.plot(train_epochs_loss_trans[0:],'-o',label="train_loss_trans")
plt.plot(val_epochs_loss_trans[0:],'-o',label="valid_loss_trans")
plt.title("epochs_loss_trans")
plt.legend()
plt.savefig('loss.png')
