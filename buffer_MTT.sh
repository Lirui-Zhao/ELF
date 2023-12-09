python -u buffer.py --CUDA_VISIBLE_DEVICES=0,1,2,3 \
--dataset=CIFAR100 --model=ConvNet \
--train_epochs=50 --num_experts=100 --save_interval=10 \
--buffer_path=buffer --data_path=data \
--num_workers=12 
