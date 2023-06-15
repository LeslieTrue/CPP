import os

import sys
sys.path.append('./')
import argparse
from tqdm import tqdm

import torch
import numpy as np
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from model.CPP_model import CPPNet
from model.sink_distance import SinkhornDistance
from data.data_utils import FeatureDataset
from loss.loss_fn import MLCLoss, TotalCodingRate
from utils import *

import torchvision
os.chdir('./metrics')
from metrics import utils

import random, string

from metrics_cluster import rect_pi_metrics, compute_numerical_rank, spectral_clustering_metrics, feature_detection,     sparsity, numerical_rank_from_singular_values
from plot import *



os.chdir('/comp_robot/daixili/tianzhe/exps/')

parser = argparse.ArgumentParser(description='CPP Training')
parser.add_argument('--hidden_dim', type=int, default=4096,
                    help='dimension of hidden state')
parser.add_argument('--z_dim', type=int, default=128,
                    help='dimension of subspace feature dimension')
parser.add_argument('--n_clusters', type=int, default=100,
                    help='number of subspace clusters to use')
parser.add_argument('--epo', type=int, default=30,
                    help='number of epochs for training')
parser.add_argument('--bs', type=int, default=2000,
                    help='input batch size for training')
parser.add_argument('--lr', type=float, default=3e-3,
                    help='learning rate (default: 0.001)')
parser.add_argument('--lr_c', type=float, default=2e-3,
                    help='learning rate (default: 0.001)')
parser.add_argument('--momo', type=float, default=0.9,
                    help='momentum (default: 0.9)')
parser.add_argument('--pigam', type=float, default=0.05,
                    help='momentum (default: 0.9)')
parser.add_argument('--wd1', type=float, default=1e-4,
                    help='weight decay for all other parameters except clustering head(default: 1e-4)')
parser.add_argument('--wd2', type=float, default=5e-3,
                    help='weight decay for clustering head (default: 5e-3)')
parser.add_argument('--eps', type=float, default=0.5,
                    help='eps squared for MCR2 objective (default: 0.1)')
parser.add_argument('--pieta', type=float, default=0.175,
                    help='temperature for gumble softmax (default: 1)')
parser.add_argument('--piiter', type=float, default=5,
                    help='temperature for gumble softmax (default: 1)')
parser.add_argument('--data_dir', type=str, default='./data/clipfeatures.pt',
                    help='path to clip feature checkpoint')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed')
parser.add_argument('--desc', type=str, default='cifar10',
                    help='description')
parser.add_argument('--warmup', type=int, default=0,
                    help='Steps of updating expansion term')
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

import clip
from PIL import Image
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = CPPNet(input_dim=768, hidden_dim = args.hidden_dim, z_dim = args.z_dim).to(device)
model = torch.nn.DataParallel(model)
model_dir = os.path.join(f'./{args.desc}')
sink_layer = SinkhornDistance(args.pieta, max_iter=args.piiter)

feature_dict = torch.load(args.data_dir)
clip_features = feature_dict['features']
clip_labels = feature_dict['ys']  

clip_features_set = FeatureDataset(clip_features, clip_labels)
train_loader = DataLoader(clip_features_set, batch_size=args.bs, shuffle=True, drop_last=True, num_workers=8)

criterion = MLCLoss(eps=0.1,gamma=1.0)
warmup_criterion = TotalCodingRate(eps = 0.1)
param_list = [p for p in model.module.pre_feature.parameters() if p.requires_grad] + [p for p in model.module.subspace.parameters() if p.requires_grad]
param_list_c = [p for p in model.module.cluster.parameters() if p.requires_grad]

optimizer = optim.SGD(param_list, lr=args.lr, momentum=args.momo, weight_decay=args.wd1,nesterov=False)
optimizerc = optim.SGD(param_list_c, lr=args.lr_c, momentum=args.momo, weight_decay=args.wd2,nesterov=False)
scaler = GradScaler()
total_wamup_steps = args.warmup
warmup_step = 0


for epoch in range(args.epo):
    with tqdm(total=len(train_loader)) as progress_bar:
        for step, (x,y) in enumerate(train_loader):
            
            x, y = x.float().to(device), y.to(device)
            
            y_np = y.detach().cpu().numpy()
            
            with autocast(enabled=True):
                z, logits = model(x)
                self_coeff = (logits @ logits.T).abs().unsqueeze(0)
                Pi = sink_layer(self_coeff)[0]
                Pi = Pi * Pi.shape[-1]
                Pi = Pi[0]
                z_list = z

                Pi_np = Pi.detach().cpu().numpy()
                
                if ((step+1) == 200 and epoch > 2 and epoch % 2==0):
                    acc_lst, nmi_lst, _, _, pred_lst = spectral_clustering_metrics(Pi_np, args.n_clusters, y_np, n_init=1)

            if warmup_step <= total_wamup_steps:
                print("warming up")
                loss = warmup_criterion(z_list)
                loss_list = [loss.item(), loss.item()]
                loss_reg = args.pigam * 0.5 * Pi.norm()**2
            else:
                loss, loss_list= criterion(z_list,Pi)
                loss_reg = args.pigam * 0.5 * Pi.norm()**2
                loss = loss + loss_reg

            optimizer.zero_grad()
            optimizerc.zero_grad()
            loss.backward()
            optimizer.step()
            optimizerc.step()

            if warmup_step == total_wamup_steps:
                print("update warmup results")
                model = update_pi_from_z(model)
            progress_bar.set_description(str(epoch))
            warmup_step += 1
            progress_bar.update(1)
            
torch.save(model.module.state_dict(), f'{model_dir}/model.pt')