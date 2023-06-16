from loss.coding_length import seg_compute
import matplotlib.pyplot as plt
from utils import plot_codinglength
import numpy as np
import clip
import torch
from metrics.clustering import spectral_clustering_metrics
import argparse
from tqdm import tqdm
from data.dataset import load_dataset
from model.CPP_model import CPPNet, CPPNet_bb
from model.sink_distance import SinkhornDistance
from torch.cuda.amp import GradScaler, autocast

parser = argparse.ArgumentParser(description='Optimal Cluster Measurement')
parser.add_argument('--hidden_dim', type=int, default=4096,
                    help='dimension of hidden state')
parser.add_argument('--z_dim', type=int, default=128,
                    help='dimension of subspace feature dimension')
parser.add_argument('--ckpt_dir', type=str, default='./ckpt.pt',
                    help='trained checkpoints')
parser.add_argument('--data_dir', type=str, default='./data',
                    help='data_dir')
parser.add_argument('--data', type=str, default='cifar10',
                    help='data')
parser.add_argument('--num_clusters', type=int, default=10,
                    help='number of clusters')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
clip_model, preprocess = clip.load("ViT-L/14", device='cpu')

model = CPPNet_bb(clip_model.visual, input_dim=768,hidden_dim=args.hidden_dim, z_dim=args.z_dim).to(device)
model = torch.nn.DataParallel(model)

sink_layer = SinkhornDistance(0.15, max_iter=5)
state_dict = torch.load(args.ckpt_dir)
model.module.load_state_dict(state_dict)

train_dataset = load_dataset(args.data, train=True, path=args.data_dir)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1500, shuffle=True, drop_last=True, num_workers=8)
logits_list = []
y_np_list = []
preds = []
datas = []
z_list = []
with tqdm(total=len(train_loader)) as progress_bar:
    for step, (x,y) in enumerate(train_loader):
        x, y = x.float().to(device), y.to(device)
        datas.append(x.detach().cpu())
        y_np_list.append(y)
        y_np = y.detach().cpu().numpy()
        with autocast(enabled=True):
            with torch.no_grad():
                z, logits = model(x)
                logits_list.append(logits)
                z_list.append(z)
            progress_bar.set_description(str(0))
            progress_bar.update(1)
        if step == 9:
            break

y_nps = torch.cat(y_np_list, dim=0).detach().cpu().numpy()
with torch.no_grad():
    logits_all = torch.cat(logits_list, dim=0).detach()
    self_coeff = (logits_all @ logits_all.T).abs().unsqueeze(0)
    Pi = sink_layer(self_coeff)[0]
    Pi = Pi * Pi.shape[-1]
    Pi = Pi[0]
    Pi_np = Pi.detach().cpu().numpy()
acc_lst, nmi_lst, _, _, pred_lst = spectral_clustering_metrics(Pi_np, args.num_clusters, y_nps)

z_features = torch.cat(z_list, dim=0).detach().cpu()
bits_list = []
for i in [5, 8, 9, 10, 11, 15, 20, 30, 50]:
    acc_lst, nmi_lst, _, _, pred_lst = spectral_clustering_metrics(Pi_np, i, y_nps)
    print(np.mean(acc_lst), np.mean(nmi_lst))
    membership = pred_lst[-1]
    num_clusters = i
    seg_bits =seg_compute(z_features, membership, num_clusters)
    bits_list.append(seg_bits)
    print(f'bits: {seg_bits}')

plot_codinglength(bits_list, [5, 8, 9, 10, 11, 15, 20, 30, 50], 'codinglength.pdf')