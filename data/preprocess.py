import torchvision
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import clip
import torch
from tqdm import tqdm
from dataset import load_dataset
import argparse

parser = argparse.ArgumentParser(description='Preprocess data into CLIP features')
parser.add_argument('--data', type=str, default='imagenet', help='dataset name, including {cifar10, cifar100coarse, cifar100, imagenet}')
parser.add_argument('--path', type=str, default='./data', help='dataset path')
parser.add_argument('--feature_dir', type=str, default='./clipfeatures.pt', help='feature path')
args = parser.parse_args()

train_dataset = load_dataset(args.data, train=True, path=args.path)
train_loader = DataLoader(train_dataset, batch_size=1500, shuffle=False, drop_last=False, num_workers=8)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
clip_model, preprocess = clip.load("ViT-L/14", device=device)
clip_model.to(device)
visual_model = torch.nn.DataParallel(clip_model)
model_dir = args.feature_dir


features = []
ys = []
with tqdm(total=len(train_loader)) as progress_bar:
    for step, (x, y) in enumerate(train_loader):
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            x_feature = visual_model.module.encode_image(x)
        
        features.append(x_feature.detach().cpu())
        ys.append(y.detach().cpu())
        progress_bar.update(1)

final_features = torch.cat(features, dim=0)
final_ys = torch.cat(ys, dim=0)
print(final_features.shape)
print(final_ys.shape)
dict = {'features': final_features, 'ys': final_ys}
torch.save(dict, model_dir)
print("done!")