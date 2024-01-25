import os

import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim

from PIL import Image
import joblib
from torchvision import transforms

import config
import utils
from dataset import CelebDataset
from generator import Generator

import argparse

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument(
    "--batch_size", type=int, default=config.batch_size, help="Batch size"
)
parser.add_argument(
    "--train_input_size",
    type=int,
    default=config.lr_size,
    help="The input size of the low resolution training images",
)
parser.add_argument(
    "--epochs",
    type=int,
    default=config.epochs,
    help="The number of epochs to train for",
)
parser.add_argument(
    "--pretraind_model_path",
    type=str,
    default=None,
    help="The path to the pre-trained model weights to initialize the generator",
)
parser.add_argument(
    "--folder_name",
    type=str,
    default="name_the_folder",
    help="Folder name to save the generated images",
)

args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

train_imgs = []
val_imgs = []
if not os.path.exists(
    os.path.join(config.data_path, "train_images.pkl")
) and not os.path.exists(os.path.join(config.data_path, "val_images.pkl")):
    all_pics = os.listdir(config.train_data_path)
    for pic_name in tqdm(all_pics):
        pic = Image.open(os.path.join(config.train_data_path, pic_name))
        if pic.size[0] >= config.min_width and pic.size[1] >= config.min_height:
            if utils.hash_image_name_to_number(pic_name) % 5 == 0:
                val_imgs.append(pic_name)
            else:
                train_imgs.append(pic_name)
    joblib.dump(train_imgs, os.path.join(config.data_path, "train_images.pkl"))
    joblib.dump(val_imgs, os.path.join(config.data_path, "val_images.pkl"))
else:
    train_imgs = joblib.load(os.path.join(config.data_path, "train_images.pkl"))[:]
    val_imgs = joblib.load(os.path.join(config.data_path, "val_images.pkl"))[:]

train_dataset = CelebDataset(
    image_list=train_imgs,
    size=args.train_input_size,
    transform=config.data_transform,
    is_train=True,
)
val_dataset = CelebDataset(
    image_list=val_imgs, size=args.train_input_size, is_train=True
)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=6
)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
)

gen = Generator().to(device)
if args.pretraind_model_path:
    print(f"Loading pre-trained model weights from {args.pretraind_model_path}")
    gen.load_state_dict(torch.load(args.pretraind_model_path))

criterion = nn.BCEWithLogitsLoss()
content_loss = utils.ContentLoss(device=device)
mse = nn.MSELoss()

resnet_mean = [0.485, 0.456, 0.406]
resnet_std = [0.229, 0.224, 0.225]
normalize = transforms.Normalize(mean=resnet_mean, std=resnet_std)

optimizer_G = optim.Adam(gen.parameters(), lr=config.lr, betas=(0.9, 0.999))

for epoch in range(args.epochs):
    tldr = tqdm(train_loader)
    lv = 0
    la = 0
    ld = 0
    for i, (low_img, high_img) in enumerate(tldr):
        gen.train()
        low_img = low_img.to(device)
        high_img = high_img.to(device)
        gen_hr_img = gen(low_img)

        lossG = mse(gen_hr_img, high_img)
        gen.zero_grad()
        lossG.backward()

        optimizer_G.step()

        lv += lossG.item()
        tldr.set_postfix(loss_resnet=0.006 * lv / (i + 1), epoch=epoch)

    ss = args.train_input_size * 4
    for i, (low_img, high_img) in enumerate((val_loader)):
        gen.eval()
        low_img = low_img.to(device)
        with torch.no_grad():
            gen_hr_img = (
                gen(low_img).to("cpu").reshape(-1, 3, ss, ss).numpy() * 0.5 + 0.5
            )
            high_img = high_img.to("cpu").reshape(-1, 3, ss, ss).numpy() * 0.5 + 0.5
            low_img = low_img.to("cpu").reshape(-1, 3, ss // 4, ss // 4).numpy()
        for ith_sample in range(args.batch_size):
            gen_img = gen_hr_img[ith_sample]
            orig_img = high_img[ith_sample]
            small_img = low_img[ith_sample]

            small_img = small_img.swapaxes(0, 1).swapaxes(1, 2)
            gen_img = gen_img.swapaxes(0, 1).swapaxes(1, 2)
            orig_img = orig_img.swapaxes(0, 1).swapaxes(1, 2)

            concatenated_image = plt.figure(figsize=(10, 5))
            plt.subplot(1, 3, 1)
            plt.imshow(small_img)
            plt.axis("off")
            plt.subplot(1, 3, 2)
            plt.imshow(gen_img)
            plt.axis("off")
            plt.subplot(1, 3, 3)
            plt.imshow(orig_img)
            plt.axis("off")
            utils.create_directory_if_not_exists(f"resgen/{args.folder_name}")
            plt.savefig(
                f"resgen/{args.folder_name}/val_image_{ith_sample}.png",
                bbox_inches="tight",
                pad_inches=0,
                format="png",
            )
            plt.close()
        break
    utils.create_directory_if_not_exists(config.model_path)
    torch.save(
        gen.state_dict(), os.path.join(config.model_path, f"gen_pretrained.pth")
    )
