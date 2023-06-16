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
parser.add_argument('--results_dir', type=str, default='./results',
                    help='folder of results')
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
clip_model, preprocess = clip.load("ViT-L/14", device=device)

if args.preprocessed:
    model = CPPNet(input_dim=768, hidden_dim = args.hidden_dim, z_dim = args.z_dim).to(device)
else:
    model = CPPNet_bb(clip_model.visual, input_dim=768, hidden_dim = args.hidden_dim, z_dim = args.z_dim).to(device)
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
clip_features_list = []
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
                clip_feature = clip_model.encode_image(x)
            clip_features_list.append(clip_feature)
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



true_labels = train_dataset.classes
txtpath = "./text_repo/imagenet_classes.txt"
with open(txtpath) as f:
    classes = f.readlines()
processed_classes = [i[1:-3] for i in classes]

txtpath = "./text_repo/fake_labels.txt"
with open(txtpath) as f:
    classes = f.readlines()
fake_classes = [i.replace("\n", "") for i in classes]
all_classes = set(true_labels + processed_classes + fake_classes)
all_classes = list(all_classes)



import torchvision
import matplotlib.pyplot as plt
fake_labels = pred_lst[-1]
data_catted = torch.cat(datas, dim=0)
clip_features = torch.cat(clip_features_list, dim=0)
text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in all_classes]).to(device)
text_features = clip_model.encode_text(text_inputs)
text_features /= text_features.norm(dim=-1, keepdim=True)
for i in range(args.num_clusters):
    print(f'cluster {i}: {np.sum(fake_labels==i)}')
    idx = np.where(fake_labels==i)[0]
    
    image_features = clip_features[idx]/torch.norm(clip_features[idx], dim=1, keepdim=True)
    image_features = image_features.to(device)
    print(image_features.shape)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    values, indices = similarity[:].topk(5)
    voting = np.zeros(len(all_classes))
    for value, index in zip(values, indices):
        index = index.detach().cpu().numpy()
        value = value.detach().cpu().numpy()
        voting[index] += value
    label_title = ''
    for label in np.argsort(-voting):
        label_title += f"{all_classes[label]}:" + " {:.2f}; ".format(voting[label])
        break
    #square figure
    plt.figure()
    plt.title(label_title,  fontdict={'fontsize': 15, 'fontweight': 'bold', 'fontfamily': 'serif'})
    cluster_imgs = data_catted[idx]
    grid = torchvision.utils.make_grid(cluster_imgs[:64], nrow=8, normalize=True)
    plt.imshow(grid.permute(1,2,0).cpu().numpy())
    plt.axis('off')
    plt.savefig(args.results_dir + '/cluster{i}.pdf', format='pdf')
    plt.close()