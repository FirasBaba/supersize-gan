import os

import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim

from PIL import Image
import joblib

from torchvision import transforms
import torch.nn.functional as F


import config
import utils
from dataset import CelebDataset
from generator import Generator
from discriminator import Discriminator

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
    "--gen_pth_path",
    type=str,
    default=None,
    help="The path to the pre-trained model weights to initialize the generator",
)
parser.add_argument(
    "--disc_pth_path",
    type=str,
    default=None,
    help="The path to the pre-trained model weights to initialize the discirminator",
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
    crop_proba=None,
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
disc = Discriminator().to(device)

if args.gen_pth_path:
    print(f"Loading pretrained generator weights from {args.gen_pth_path}")
    gen.load_state_dict(torch.load(args.gen_pth_path))
if args.disc_pth_path:
    print(f"Loading pretrained discriminator weights from {args.disc_pth_path}")
    disc.load_state_dict(torch.load(args.disc_pth_path))

criterion = nn.BCEWithLogitsLoss()
content_loss = utils.ContentLoss(device=device)
kl_loss = torch.nn.KLDivLoss(reduction="batchmean")

optimizer_G = optim.Adam(gen.parameters(), lr=config.lr, betas=(0.9, 0.999))
optimizer_D = optim.Adam(disc.parameters(), lr=config.lr, betas=(0.9, 0.999))

scheduler_G = optim.lr_scheduler.MultiStepLR(
    optimizer_G, milestones=[6, 12, 18], gamma=0.5
)
scheduler_D = optim.lr_scheduler.MultiStepLR(
    optimizer_D, milestones=[6, 12, 18], gamma=0.5
)


resnet_mean = [0.485, 0.456, 0.406]
resnet_std = [0.229, 0.224, 0.225]
normalize = transforms.Normalize(mean=resnet_mean, std=resnet_std)

# Training loop
for epoch in range(args.epochs):
    tldr = tqdm(train_loader)
    lv = 0
    la = 0
    ld = 0
    lk = 0
    print(f"Learning rate: {optimizer_D.param_groups[0]['lr']}")
    for i, (low_img, high_img) in enumerate(tldr):
        gen.train()
        disc.train()
        low_img = low_img.to(device)
        high_img = high_img.to(device)

        gen_hr_img = gen(low_img)
        disc_original = disc(high_img)

        lossD_original = criterion(
            disc_original,
            torch.ones_like(disc_original) - 0.1 * torch.rand_like(disc_original),
        )

        disc_gen_hr = disc(gen_hr_img)
        lossD_hen_hr = criterion(disc_gen_hr, torch.zeros_like(disc_gen_hr))

        lossD = lossD_original + lossD_hen_hr

        optimizer_D.zero_grad()
        lossD.backward(retain_graph=True)
        optimizer_D.step()

        # Training the Generator
        output = disc(gen_hr_img)

        content_generated = gen_hr_img * 125 + 125
        content_original = high_img * 125 + 125

        gen_hr_img_normalized = normalize(content_generated)
        high_img_normalized = normalize(content_original)

        loss_vgg = 0.006 * content_loss(gen_hr_img_normalized, high_img_normalized)
        loss_adv = 1e-3 * criterion(output, torch.ones_like(output))

        lossG = loss_adv + loss_vgg
        optimizer_G.zero_grad()
        lossG.backward()
        optimizer_G.step()

        lv += loss_vgg.item()
        la += loss_adv.item()
        ld += lossD.item()
        tldr.set_postfix(
            epoch=epoch,
            loss_vgg=lv / (i + 1),
            loss_adv=la / (i + 1),
            lossD=ld / (i + 1),
        )
    high_res_size = args.train_input_size * 4

    for i, (low_img, high_img) in enumerate(tqdm(val_loader)):
        gen.eval()
        low_img = low_img.to(device)
        with torch.no_grad():
            gen_hr_img = (
                gen(low_img)
                .to("cpu")
                .reshape(-1, 3, args.train_input_size * 4, args.train_input_size * 4)
                .numpy()
                * 0.5
                + 0.5
            )
            high_img = (
                high_img.to("cpu")
                .reshape(-1, 3, args.train_input_size * 4, args.train_input_size * 4)
                .numpy()
                * 0.5
                + 0.5
            )
            low_img = (
                low_img.to("cpu")
                .reshape(-1, 3, args.train_input_size, args.train_input_size)
                .numpy()
            )
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
            utils.create_directory_if_not_exists(f"results/{args.folder_name}")
            plt.savefig(
                f"results/{args.folder_name}/val_image_{ith_sample}.png",
                bbox_inches="tight",
                pad_inches=0,
                format="png",
            )
            plt.close()
        break
    utils.create_directory_if_not_exists(config.model_path)
    torch.save(gen.state_dict(), os.path.join(config.model_path, f"gen.pth"))
    torch.save(disc.state_dict(), os.path.join(config.model_path, f"disc.pth"))
    scheduler_G.step()
    scheduler_D.step()
