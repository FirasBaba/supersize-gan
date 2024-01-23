import os
import hashlib
import torch
import torch.nn as nn
import torchvision.models as models

def hash_image_name_to_number(image_name):
    hash_object = hashlib.sha256()
    image_name_bytes = image_name.encode('utf-8')
    hash_object.update(image_name_bytes)
    hashed_number = int(hash_object.hexdigest(), 16)
    return hashed_number

def create_directory_if_not_exists(directory_path):
    if not os.path.exists(directory_path):
        print(f"Creating directory: {directory_path}")
        os.makedirs(directory_path)

class ContentLoss(nn.Module):
    def __init__(self, device, model="resnet"):
        super(ContentLoss, self).__init__()
        if model is None:
            self.model = models.vgg19(pretrained=True)
            self.model = self.model.features[:36].eval().to(device)
        else:
            self.model = models.resnet18(pretrained=True)
            self.model = torch.nn.Sequential(*list(self.model.children())[:9])
            self.model = self.model.eval().to(device)
        self.distance_loss = nn.MSELoss()

        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, img1, img2):
        # Apply the specified function to the input images
        embd1 = self.model(img1)
        embd2 = self.model(img2)
        # Compute the L2 distance between the processed images
        l2_distance = self.distance_loss(embd1, embd2)

        return l2_distance