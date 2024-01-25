# SRGAN-CryptoPunk

## Overview

SRGAN-CryptoPunk is a project inspired by the SRGAN paper, implementing a modified version of the SRGAN architecture. The primary goal of this project is to provide a simple and fun implementation while refreshing my legacy knowledge in GANs. The model takes an image as input and scales it by a factor of x4, resulting in an image with 16 times more pixels.

## Repository Files

- **config.py**: Configurations of the pipeline.
- **dataset.py**: PyTorch class for the customized dataset.
- **discriminator.py**: Discriminator class.
- **generator.py**: Generator class.
- **train_generator.py**: Pretrain the model using mean squared error to avoid getting stuck in local minima when training the final GAN.
- **train.py**: Train the discriminator and the generator together.
- **utils.py**: Utility classes and functions.

## How to Use

1. Clone the repository:
```
git clone https://github.com/FirasBaba/srgan-cryptopunk.git
```

2. Navigate to the project directory:
```
cd srgan-cryptopunk
```

3. Create a new folder for data:
```
mkdir data
```


4. Download your data and place it under the `data/` folder.

5. Update the training data path in the config file (`config.py`):
```
train_data_path= "../data/your_data_folder_name"
```

6. Build the Docker image:
```
make build
```
7. Run the Docker container:
```
make run
```
8. Execute the pipeline inside the Docker container:
```
sh pipeline.sh
```

Feel free to customize the `pipeline.sh` file for further training epochs or bigger/smaller images or batch sizes.

## Dataset
The training data for this project is sourced from the CelebA dataset, a widely used dataset for face attribute analysis. The CelebA dataset comprises over 200,000 celebrity images annotated with 40 attribute labels.

### CelebA Dataset Information

- **Dataset Name**: CelebA
- **Number of Images**: Over 200,000
- **Resolution**: Varied
##### [CelebA dataset: Download link](https://drive.google.com/drive/folders/0B7EVK8r0v71peklHb0pGdDl6R28?resourcekey=0-f5cwz-nTIQC3KsBn3wFn7A)

## Training Techniques

I employed progressive training to iterate faster in this project. The generator is initially trained with a resolution of 12x12, then 24x24, 48x48, and finally 96x96. The learning rate is reset at each new iteration. After this, the pretrained weights are used to train the final GAN. The discriminator is trained with binary cross-entropy loss, while the generator is trained with a content loss and adversarial/BCE loss. Both losses are linearly weighted, following the principles outlined in the SRGAN paper. The content loss uses ResNet18 to extract embedding features from the high-resolution image and the target image, deviating from the original VGG approach in the SRGAN paper.

## Citation
- Christian Ledig, Lucas Theis, Ferenc Huszar, Jose Caballero, Andrew Cunningham, Alejandro Acosta, Andrew Aitken, Alykhan Tejani, Johannes Totz, Zehan Wang, Wenzhe Shi. **Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network**, 2016.
[Link to the paper](https://arxiv.org/abs/1609.04802)
```
@InProceedings{srgan,
    author = {Christian Ledig, Lucas Theis, Ferenc Huszar, Jose Caballero, Andrew Cunningham, Alejandro Acosta, Andrew Aitken, Alykhan Tejani, Johannes Totz, Zehan Wang, Wenzhe Shi},
    title = {Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network},
    booktitle = {arXiv},
    year = {2016}
}
```
- Liu, Ziwei and Luo, Ping and Wang, Xiaogang and Tang, Xiaoou. **Deep Learning Face Attributes in the Wild**, 2015.
```
@inproceedings{liu2015faceattributes,
  title = {Deep Learning Face Attributes in the Wild},
  author = {Liu, Ziwei and Luo, Ping and Wang, Xiaogang and Tang, Xiaoou},
  booktitle = {Proceedings of International Conference on Computer Vision (ICCV)},
  month = {December},
  year = {2015} 
}
```


