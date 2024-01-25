from torchvision import transforms

data_path = "../data"
train_data_path = "../data/img_celeba"
model_path = "weights"

min_width = 224
min_height = 224

lr_size = 24

batch_size = 48
epochs = 100
lr = 1e-4

crop_probability = 0.01

data_transform = transforms.Compose(
    [
        transforms.RandomRotation(
            degrees=30
        ),
        transforms.RandomAffine(
            degrees=0, translate=(0.1, 0.1)
        ),
        transforms.RandomHorizontalFlip(),
    ]
)
