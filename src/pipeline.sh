# python3 train_generator.py --train_input_size 12 --epochs 10 --folder_name size12
# python3 train_generator.py --train_input_size 24 --epochs 10 --pretraind_model_path weights/gen_pretrained.pth --folder_name size24
# python3 train_generator.py --train_input_size 48 --epochs 4 --pretraind_model_path weights/gen_pretrained.pth --folder_name size48
# python3 train_generator.py --batch_size 16 --train_input_size 96 --epochs 1 --pretraind_model_path weights/gen_pretrained.pth --folder_name size96

# python3 train.py --batch_size 48\
#                  --train_input_size 24\
#                  --epochs 20\
#                  --gen_pth_path weights/gen_pretrained.pth\
#                  --folder_name srgan_size24

# python3 train.py --batch_size 16\
#                  --train_input_size 48\
#                  --epochs 20\
#                  --gen_pth_path weights/gen.pth\
#                  --disc_pth_path weights/disc.pth\
#                  --folder_name srgan_size48

python3 train.py --batch_size 12\
                 --train_input_size 96\
                 --epochs 10\
                 --gen_pth_path weights/gen.pth\
                 --disc_pth_path weights/disc.pth\
                 --folder_name srgan_size96

