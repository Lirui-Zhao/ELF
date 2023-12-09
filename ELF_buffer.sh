python -u buffer.py --CUDA_VISIBLE_DEVICES=0,1,2,3 \
--dataset=CIFAR100 --model=ConvNetW512 \
--train_epochs=200 --num_experts=1 --save_interval=1 \
--buffer_path=buffer --data_path=data \
--num_workers=12 \
--mom=0.9 --l2=0.0005 --decay 
