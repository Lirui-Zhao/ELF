python -u distill_DSA.py  --dataset CIFAR100  --model ConvNet  --ipc 10 \
--init real --method DSA --dsa_strategy color_crop_cutout_flip_scale_rotate \
--num_eval 5 --num_exp 1 --dis_metric mse \
--CUDA_VISIBLE_DEVICES 0,1,2,3 --data_path data