python -u distill_DM.py  --dataset CIFAR100  --model ConvNet  --ipc 10  \
--dsa_strategy color_crop_cutout_flip_scale_rotate  \
--init real  --lr_img 1  --num_exp 1  --num_eval 5 \
--CUDA_VISIBLE_DEVICES 0,1,2,3 --data_path data
# Empirically, for CIFAR10 dataset we set --lr_img 1 for --ipc = 1/10/50, --lr_img 10 for --ipc = 100/200/500/1000/1250. For CIFAR100 dataset, we set --lr_img 1 for --ipc = 1/10/50/100/125.