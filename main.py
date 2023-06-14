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
# from model.clustering_layer import ClusteringViT_parallel_v4 as Clustering
# from model.clustering_layer import SinkhornDistance
# from model.clustering_layer import tinyClusteringLayer
# from model.clustering_layer import SubspaceClusterNetworkCLIP2
# from architectures.models_se import SubspaceClusterNetworkSepSE as SubspaceClusterNetwork
from data.dataset import load_dataset
from data.clip_feature import FeatureDataset, get_features
from loss.loss_fn import MaximalCodingRateReduction, TotalCodingRate
import torchvision
os.chdir('./metrics')
# from func import chunk_avg, cluster_acc

from metrics import utils

import random, string

from metrics_cluster import rect_pi_metrics, compute_numerical_rank, spectral_clustering_metrics, feature_detection,     sparsity, numerical_rank_from_singular_values
from plot import *

from utils_my import *

from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode
BICUBIC = InterpolationMode.BICUBIC
my_preprocess = Compose([
        Resize(224, interpolation=BICUBIC),
        CenterCrop(224),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


os.chdir('/comp_robot/daixili/tianzhe/exps/')

parser = argparse.ArgumentParser(description='Unsupervised Learning')
parser.add_argument('--arch', type=str, default='resnet18cifar',
                    help='architecture for deep neural network (default: resnet18cifar)')
parser.add_argument('--z_dim', type=int, default=128,
                    help='dimension of subspace feature dimension (default: 64)')
parser.add_argument('--n_clusters', type=int, default=100,
                    help='number of subspace clusters to use (default: 10)')
parser.add_argument('--data', type=str, default='cifar10',
                    help='dataset used for training (default: cifar10)')
parser.add_argument('--aug_name', type=str, default='cifar10_sup',
                    help='name of augmentation to use')
parser.add_argument('--epo', type=int, default=30,
                    help='number of epochs for training (default: 100)')
parser.add_argument('--load_epo', type=int, default=600,
                    help='epo to load pre-trained checkpoint from')
parser.add_argument('--train_backbone', action='store_true',
                    help='whether to also train parameters in backbone')
parser.add_argument('--validate_every', type=int, default=10,
                    help='validate clustering accuracy every this epochs and save results (default: 10)')
parser.add_argument('--bs', type=int, default=2000,
                    help='input batch size for training (default: 1000)')
parser.add_argument('--n_views', type=int, default=1,
                    help='number of augmentations per sample')
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
parser.add_argument('--tau', type=float, default=1,
                    help='temperature for gumble softmax (default: 1)')
parser.add_argument('--pieta', type=float, default=0.175,
                    help='temperature for gumble softmax (default: 1)')
parser.add_argument('--piiter', type=float, default=5,
                    help='temperature for gumble softmax (default: 1)')
parser.add_argument('--z_weight', type=float, default=100.,
                    help='weight for z_sim loss (default: 100)')
parser.add_argument('--doc', type=str, default='',
                    help='extra information to add to folder name')
parser.add_argument('--save_dir', type=str, default='./exps/',
                    help='base directory for saving PyTorch model. (default: ./exps/)')
parser.add_argument('--data_dir', type=str, default='../../data/',
                    help='path to dataset folder')
parser.add_argument('--gpu_ids', default=[0,1,5,6], type=eval, 
                    help='IDs of GPUs to use')
parser.add_argument('--fp16', action='store_true',
                    help='Whether or not to use 16-bit precision GPU training.')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed')
parser.add_argument('--desc', type=str, default='cifar100',
                    help='description')
parser.add_argument('--input_dim', type=int, default=768,
                    help='random seed')
parser.add_argument('--my_z_dim', type=int, default=768,
                    help='random seed')
parser.add_argument('--warmup', type=int, default=0,
                    help='random seed')
parser.add_argument('--hidden_dim', type=int, default=4096,
                    help='random seed')
args = parser.parse_args()

wandb.init(project="CLIP-Clustering-CIFAR100", entity="mcr2", name=f'testCLIP'+args.desc)

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

import clip
from PIL import Image

# device = 'cuda:'+ str(args.gpu_ids[0])
device = 'cuda'
# clip_model, preprocess = clip.load("ViT-L/14", device=device)

# model = Clustering(clip_model.visual,input_dim=args.input_dim, z_dim = args.my_z_dim).to(device)
# cluster_layer = tinyClusteringLayer(input_dim=args.input_dim, z_dim = args.my_z_dim).to(device)

model = Clustering(input_dim=768, hidden_dim = args.hidden_dim, z_dim = args.z_dim).to(device)
model = torch.nn.DataParallel(model)
model_dir = os.path.join(f'./{args.desc}')
sink_layer = SinkhornDistance(args.pieta, max_iter=args.piiter)
os.makedirs(model_dir, exist_ok = True)
os.makedirs(model_dir + '/cluster_imgs/', exist_ok=True)
os.makedirs(model_dir + '/pca_figures/', exist_ok=True)
os.makedirs(model_dir + '/checkpoints/', exist_ok=True)

# data_path = "/comp_robot/cv_public_dataset/imagenet1k/train/"
# # train_dataset = load_dataset('cifar100', 'cifar_simclr_norm', contrastive=True if args.n_views>1 else False,n_views=args.n_views,path=data_path)
# # test_dataset = load_dataset('cifar100','cifar_simclr_norm',use_baseline=True,train=True,contrastive=False,path=data_path)
# # train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, drop_last=True, num_workers=8)
# # test_loader = DataLoader(test_dataset, batch_size=args.bs, shuffle=True, drop_last=False, num_workers=8)
# train_dataset = torchvision.datasets.ImageFolder(root="/comp_robot/cv_public_dataset/imagenet1k/train/",transform=my_preprocess)
# valid_dataset = torchvision.datasets.ImageFolder(root="/comp_robot/cv_public_dataset/imagenet1k/val/",transform=my_preprocess)
# train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, drop_last=True, num_workers=8)
# clip_features, clip_labels = get_features(train_dataset, clip_model, device)
feature_dict = torch.load('/comp_robot/daixili/tianzhe/clipfeatures.pt')
clip_features = feature_dict['features']
clip_labels = feature_dict['ys']  
print("Feature shape:", clip_features.shape)
clip_feature_set = FeatureDataset(clip_features, clip_labels)
train_loader = DataLoader(clip_feature_set, batch_size=args.bs, shuffle=True, drop_last=True, num_workers=8)
# x = train_dataset[0][0]
# print("Test feature shape:", x[0].shape)
criterion = MaximalCodingRateReduction(eps=0.1,gamma=1.0)
warmup_criterion = TotalCodingRate(eps = 0.1)
param_list = [p for p in model.module.pre_feature.parameters() if p.requires_grad] + [p for p in model.module.subspace.parameters() if p.requires_grad]
param_list_c = [p for p in model.module.cluster.parameters() if p.requires_grad]

optimizer = optim.SGD(param_list, lr=args.lr, momentum=args.momo, weight_decay=args.wd1,nesterov=False)
optimizerc = optim.SGD(param_list_c, lr=args.lr_c, momentum=args.momo, weight_decay=args.wd2,nesterov=False)
scaler = GradScaler()
total_wamup_steps = args.warmup
warmup_step = 0

def update_pi_from_z(net):
    import copy
    model_dict = net.state_dict()
    save_dict = copy.deepcopy(model_dict)
    to_rename_keys = []
    for key in save_dict:
    #     if 'cluster' in key:
    #         to_del_keys.append(key)
        if 'subspace' in key:
            to_rename_keys.append(key)

    # for key in to_del_keys:
    #     del save_dict[key]
    #     print(f'deleted key {key}')

    for key in to_rename_keys:
        print(f'renamed key {key}')
        pre, post = key.split('subspace')
        save_dict[pre + 'cluster' + post] = save_dict.pop(key)

    model_dict.update(save_dict)
    log = net.load_state_dict(model_dict)
    print(log)
    return net
for epoch in range(args.epo):
    # wandb.log({"epoch": epoch, "lr_z": optimizer.param_groups[0]['lr'], "lr_c": optimizerc.param_groups[0]['lr']})
    with tqdm(total=len(train_loader)) as progress_bar:
        for step, (x,y) in enumerate(train_loader):
            # x = torch.cat(x,dim=0)
            warmup_step += 1
            x, y = x.float().to(device), y.to(device)
            
            y_np = y.detach().cpu().numpy()
            
            with autocast(enabled=True):
                # print(x.type())
                z, logits = model(x)
                logits = z
                self_coeff = (logits @ logits.T).abs().unsqueeze(0)
                Pi = sink_layer(self_coeff)[0]
                Pi = Pi * Pi.shape[-1]
                # print(f'Pi.shape: {Pi.shape}')
                # Pi = (Pi[0] + Pi[1]) / 2
                Pi = Pi[0]
                # print(Pi.type(), z.type())
                # print(f'Pi.shape: {Pi.shape}')
                z_list = z
                # print(f'z_list.shape: {z_list.shape}')
                # metrics on z
                # z_list = z.chunk(2, dim=0)
                # z_sim = (z_list[0] * z_list[1]).sum(1).mean()

                # sim_mat = z_list[0] @ z_list[1].T
                # sim_mat = sim_mat.detach()
                # spe, nnz = metrics(sim_mat, y_np)
                # wandb.log({"z_spe": spe, "z_nnz": nnz})
                
                # metrics on pi
                Pi_np = Pi.detach().cpu().numpy()
                pi_spe = feature_detection(Pi_np, y_np)   
                nnz_2 = sparsity(Pi_np, 1e-2)
                nnz_3 = sparsity(Pi_np, 1e-3)
                nnz_4 = sparsity(Pi_np, 1e-4)
                nnz_5 = sparsity(Pi_np, 1e-5)
                nnz_6 = sparsity(Pi_np, 1e-6)
                
                wandb.log({"pi_spe": pi_spe, "pi_nnz_2": nnz_2, "pi_nnz_3": nnz_3, 
                           "pi_nnz_4": nnz_4, "pi_nnz_5": nnz_5, "pi_nnz_6": nnz_6})
                
                # acc_lst, nmi_lst, _, _, pred_lst = spectral_clustering_metrics(Pi_np, args.n_clusters, y_np)
                #  pred_order = np.argsort(pred_lst[-1])
                wandb.log({ "pi_nnz_3": nnz_3, 
                           "pi_nnz_4": nnz_4, "pi_nnz_5": nnz_5, "pi_nnz_6": nnz_6})
                if (step+1)%5 == 0:
                    plot_fn = f'epoch{epoch}_batch{step}'
                    plot_tit = f' epoch:{epoch} batch:{step}'
                    plot_heatmap(model_dir, z_list.detach().cpu().float(), y_np, 10, plot_fn, title='ZtZ'+plot_tit)
                    plot_membership(model_dir, Pi_np, y_np, plot_fn+'_1', title=f'Pi'+plot_tit, vmax=0.1)
                    plot_membership(model_dir, Pi_np, y_np, plot_fn+'_2', title=f'Pi'+plot_tit, vmax=0.01)
                    plot_membership(model_dir, Pi_np, y_np, plot_fn+'_3', title=f'Pi'+plot_tit, vmax=0.001)
                if ((step+1) == 200 and epoch > 2 and epoch % 2==0):
                    acc_lst, nmi_lst, _, _, pred_lst = spectral_clustering_metrics(Pi_np, args.n_clusters, y_np, n_init=1)
                    wandb.log({"pi acc": acc_lst[-1], "nmi": nmi_lst[-1]})
                if ((step+1) == 200 and epoch >2 and epoch % 2==0):
                    torch.save(model.module.state_dict(), f'{model_dir}/model_{epoch}_{step}.pt')

                # if 2>1:
                #     z_avg = chunk_avg(z,n_chunks=2,normalize=True)

                #     rank_lst = compute_numerical_rank(z_avg.detach().cpu().numpy(), y.detach().cpu().numpy(),
                #                                       10, tau=0.95)
                #     wandb.log({f'z_rank_class{c - 1}': r for c, r in enumerate(rank_lst)})
            
            if warmup_step <= total_wamup_steps:
                print("warming up")
                loss = warmup_criterion(z_list)
                loss_list = [loss.item(), loss.item()]
                loss_reg = args.pigam * 0.5 * Pi.norm()**2
            else:
                loss, loss_list= criterion(z_list,Pi,num_classes=args.bs)
                loss_reg = args.pigam * 0.5 * Pi.norm()**2
                loss = loss + loss_reg
            # loss_list += [z_sim.item()]
            # print(f'loss_list: {loss_list}')
            optimizer.zero_grad()
            optimizerc.zero_grad()
            if False:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizerc.step()
            else:
                loss.backward()
                optimizer.step()
                optimizerc.step()

            wandb.log({"loss_dR": -loss_list[0] + loss_list[1], "loss_R": loss_list[0],
                       "loss_Rc": loss_list[1], "loss_reg": loss_reg.item(),
                       "loss_all": loss.item()})
            if warmup_step == total_wamup_steps:
                print("update warmup results")
                model = update_pi_from_z(model)
            progress_bar.set_description(str(epoch))
            # progress_bar.set_postfix(loss= -loss_list[0] + loss_list[1],
            #                          loss_d=loss_list[0],
            #                          loss_c=loss_list[1],
            #                          z_sim=z_sim.item()
            #                         )
            progress_bar.update(1)
        # acc_lst, nmi_lst, _, _, pred_lst = spectral_clustering_metrics(Pi_np, args.n_clusters, y_np)
        # wandb.log({"pi acc": acc_lst[-1], "nmi": nmi_lst[-1]})
#     if (epoch+1)%args.validate_every==0:
#         save_name_img = model_dir + '/cluster_imgs/cluster_imgs_ep' + str(epoch+1)
#         save_name_fig = model_dir + '/pca_figures/z_space_pca' + str(epoch+1)
# #         acc_single, acc_merge, NMI, ARI = cluster_acc(test_loader,net,device,print_result=True,save_name_img=save_name_img,save_name_fig=save_name_fig)
# #         wandb.log({"train_acc_single": acc_single, "train_acc_merge": acc_merge, "train_nmi": NMI, "train_ari": ARI})
#         utils.save_ckpt(model_dir, net, optimizer, optimizerc, epoch + 1 + args.load_epo)

torch.save(model.module.state_dict(), f'{model_dir}/model.pt')