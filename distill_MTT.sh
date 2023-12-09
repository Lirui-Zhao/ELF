python -u distill_MTT.py --CUDA_VISIBLE_DEVICES=0,1,2,3 \
--model=ConvNet --dataset=CIFAR100 --ipc=10 \
--syn_steps=20 --expert_epochs=2 --max_start_epoch=40 \
--lr_img=1e3 --lr_lr=1e-5 --lr_teacher=1e-2 \
--Iteration=5000 --load_all \
--buffer_path=buffer --data_path=data \
