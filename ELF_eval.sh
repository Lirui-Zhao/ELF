python -u ELF_eval.py --CUDA_VISIBLE_DEVICES=0,1,2,3 \
--dataset=CIFAR100 --ipc=10 \
--loss_mode=front_rear_task \
--distill_model=ConvNet \
--feature_model=ConvNetW512_L3 \
--eval_model=ResNet18_Layered --eval_mode=itself \
--feature_epochs=100 --buffer_mom \
--lamda_front=1 --lamda_rear=1 --feature_loss_mode=CE \
--data_path=data --distilled_data_dir=result_MTT \
--buffer_path=buffer 
