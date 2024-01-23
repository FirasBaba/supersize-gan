from torchvision import transforms
data_path = "../data" 
train_data_path = "../data/img_celeba"
cryptopunk_path = "../data/cryptopunks/imgs/imgs"
model_path = "weights"

lr_size = 24

batch_size = 48 #224
epochs = 100
lr = 1e-4 #1e-4

crop_probability = 0.01

data_transform = transforms.Compose([
    transforms.RandomRotation(degrees=30),  # Random rotation between -30 and 30 degrees
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Random translation by 10% of image size
    transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
    ])
