import os
import sys
import argparse
from datetime import datetime
import copy

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from utils import (
    get_dataset,
    get_network,
    get_daparam,
    get_eval_pool,
    TensorDataset,
    evaluate_synset,
    epoch,
    evaluate_feature_synset,
    ParamDiffAug,
    seed_torch
)
from reparam_module import ReparamModule


import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.CUDA_VISIBLE_DEVICES
    seed_torch(args.seed)
    args.dsa = True if args.dsa == 'True' else False
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dsa_param = ParamDiffAug()
    args.distributed = torch.cuda.device_count() > 1
    
    args.distilled_data_dir = os.path.join(args.distilled_data_dir, args.dataset)
    if args.dataset in ["CIFAR10", "CIFAR100"] and not args.zca:
        args.distilled_data_dir += "_NO_ZCA"
    args.distilled_data_dir=os.path.join(args.distilled_data_dir,args.distill_model.split('_')[0],str(args.ipc)+'ipc')
    if args.distill_loss:
        args.distilled_data_dir=os.path.join(args.distilled_data_dir,args.distill_loss)
    if args.distilled_time:
        args.distilled_data_dir=os.path.join(args.distilled_data_dir,args.distilled_time)
    assert os.path.isdir(args.distilled_data_dir), f'Error: no eval directory found!:{args.distilled_data_dir}'
    
    
    images_best = torch.Tensor(torch.load(args.distilled_data_dir + '/images_best.pt')).to(args.device)
    labels_best = torch.load(args.distilled_data_dir + '/labels_best.pt').to(args.device)
    
    now = datetime.now()
    current_time = now.strftime("%m-%d-%Y-%H:%M:%S")
    print(f'current time: \n{current_time}')
    print('Hyper-parameters: \n', args.__dict__)
    
    (
        channel,
        im_size,
        num_classes,
        class_names,
        mean,
        std,
        dst_train,
        dst_test,
        testloader,
        loader_train_dict,
        class_map,
        class_map_inv,
    ) = get_dataset(
        args.dataset, args.data_path, args.batch_real, args.subset, args=args
    )
    
    feature_best=torch.zeros(len(images_best),1)
    if args.loss_mode != 'task':
        expert_params_path = os.path.join(args.buffer_path, args.dataset)
        if args.dataset in ["CIFAR10", "CIFAR100"] and not args.zca:
            expert_params_path += "_NO_ZCA"
        expert_params_path = os.path.join(expert_params_path, args.feature_model.split('_')[0])
        if args.dataset == 'ImageNet':
            expert_params_path=os.path.join(expert_params_path,'lr_0.01','epoch_200')
        if args.buffer_mom:
            expert_params_path=os.path.join(expert_params_path,'momentum')
        expert_params_path= os.path.join(expert_params_path,'replay_buffer_0.pt')
        assert os.path.isfile(expert_params_path), f'Error: no buffer file found!:{expert_params_path}'

        expert_params=torch.load(expert_params_path)[-1][args.feature_epochs]
        print(f'load feature params:{expert_params_path} epoch:{args.feature_epochs}')


        expert_params=torch.cat([p.data.reshape(-1) for p in expert_params], 0).to(args.device)
        
        expert_net = get_network(args.feature_model, channel, num_classes, im_size, dist=False,args=args).to(
            args.device
        )
        print(f'get feature network:{args.feature_model}')
        expert_net = ReparamModule(expert_net).eval()
        if args.distributed:
            expert_params = (
                expert_params.unsqueeze(0).expand(torch.cuda.device_count(), -1)
            )
            expert_net = torch.nn.DataParallel(expert_net)
        
        feature_best=[]
        if args.batch_syn is None:
            args.batch_syn = num_classes * args.ipc
        indices = torch.arange(len(images_best))
        indices_chunks = list(torch.split(indices, args.batch_syn))
        for these_indices in indices_chunks: 
            images_best_part = images_best[these_indices]
            if args.feature_model.endswith('US_feature'):
                feature_best_part,_ = expert_net(images_best_part,flat_param=expert_params,width_mult=args.width_mult)
            else:
                feature_best_part,_ = expert_net(images_best_part,flat_param=expert_params)
            feature_best.append(feature_best_part)
        feature_best=torch.cat(feature_best,dim=0).float()

    model_eval_pool = get_eval_pool(args.eval_mode, args.eval_model)
    for model_eval in model_eval_pool:
        print(
            '-------------------------\nEvaluation\nmodel_distill = %s,model_feature = %s,model_eval = %s'
            % (args.distill_model,args.feature_model, model_eval)
        )
        print(f'loss mode:{args.loss_mode}')
        print('-------------------------')

        if args.dsa:
            print('DSA augmentation strategy: \n', args.dsa_strategy)
            print('DSA augmentation parameters: \n', args.dsa_param.__dict__)
        else:
            print('DC augmentation parameters: \n', args.dc_aug_param)
        acc_test_list = []
        for it_eval in range(args.num_eval):
            net_eval = get_network(model_eval, channel, num_classes, im_size, args=args).to(args.device)  # get a random model
            _, acc_train, acc_test=evaluate_feature_synset(
                it_eval,
                net_eval,
                images_best,
                feature_best,
                labels_best,
                testloader,
                args,
            )
            acc_test_list.append(acc_test)
        acc_test_list=np.array(acc_test_list)
        acc_test_mean=np.mean(acc_test_list)
        acc_test_std=np.std(acc_test_list)
        print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------'
                    % (len(acc_test_list), model_eval, acc_test_mean, acc_test_std))
        print('Evaluation Model = %s; ACC = %.2f$\pm$%.2f'% (model_eval, acc_test_mean*100, acc_test_std*100))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument(
        '--subset',
        type=str,
        default='imagenette',
        help='ImageNet subset. This only does anything when --dataset=ImageNet',
    )
    parser.add_argument('--distill_model', type=str, default='ConvNet', help="fistilled data's model ")
    parser.add_argument('--ipc', type=int, default=1, help='image(s) per class')
    parser.add_argument('--loss_mode', type=str, choices=["task", "front_rear_task","front_task", "rear_task", "front_rear"],default='front_rear_task', help='loss mode, full means MTT, before_after_full means ours method')
    parser.add_argument('--feature_loss_mode', type=str, choices=["L2","CE", "L1","COS"], default='CE', help='feature loss mode')
    
    parser.add_argument('--eval_model', type=str, default='ConvNet', help='model to eval')
    parser.add_argument('--feature_model', type=str, default='ConvNet', help='model to get feature')
    parser.add_argument('--feature_epochs', type=int, default=50)
    
    parser.add_argument(
        '--eval_mode',
        type=str,
        default='S',
        help='eval_mode, check utils.py for more info',
    )
    parser.add_argument(
        '--num_eval', type=int, default=5, help='how many networks to evaluate on'
    )
    parser.add_argument(
        '--eval_it', type=int, default=100, help='how often to evaluate'
    )
    parser.add_argument(
        '--epoch_eval_train',
        type=int,
        default=1000,
        help='epochs to train a model with synthetic data',
    )
    parser.add_argument(
        '--Iteration',
        type=int,
        default=5000,
        help='how many distillation steps to perform',
    )
    parser.add_argument(
        '--lr_net',
        type=float,
        default=0.01,
        help='initialization for synthetic learning rate',
    )
    parser.add_argument(
        '--lamda_front',
        type=float,
        default=1,
        help='hyperparameter lamda of front loss',
    )
    parser.add_argument(
        '--lamda_rear',
        type=float,
        default=1,
        help='hyperparameter lamda of rear loss',
    )

    parser.add_argument(
        '--batch_real', type=int, default=256, help='batch size for real data'
    )
    parser.add_argument(
        '--batch_syn',
        type=int,
        default=None,
        help='should only use this if you run out of VRAM',
    )
    parser.add_argument(
        '--batch_train', type=int, default=256, help='batch size for training networks'
    )

    parser.add_argument(
        '--pix_init',
        type=str,
        default='real',
        choices=["noise", "real"],
        help='noise/real: initialize synthetic images from random noise or randomly sampled real images.',
    )

    parser.add_argument(
        '--dsa',
        type=str,
        default='True',
        choices=['True', 'False'],
        help='whether to use differentiable Siamese augmentation.',
    )
    parser.add_argument(
        '--dsa_strategy',
        type=str,
        default='color_crop_cutout_flip_scale_rotate',
        help='differentiable Siamese augmentation strategy',
    )

    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument(
        '--buffer_path', type=str, default='./buffers', help='buffer path'
    )
    parser.add_argument('--buffer_mom', action='store_true', help="buffer has momentum")
    parser.add_argument(
        '--distilled_data_dir', type=str, default=None, help='eval distilled dataset dir path'
    )

    parser.add_argument(
        '--expert_epochs',
        type=int,
        default=3,
        help='how many expert epochs the target params are',
    )
    parser.add_argument(
        '--syn_steps',
        type=int,
        default=20,
        help='how many steps to take on synthetic data',
    )
    parser.add_argument(
        '--max_start_epoch', type=int, default=25, help='max epoch we can start at'
    )

    parser.add_argument('--zca', action='store_true', help="do ZCA whitening")
    

    parser.add_argument(
        '--no_aug',
        type=bool,
        default=False,
        help='this turns off diff aug during distillation',
    )
    parser.add_argument(
        '--transforms_normalize_syn',
        action='store_true',
        help='transforms_normalize_syn',
    )
    parser.add_argument(
        '--CUDA_VISIBLE_DEVICES',
        type=str,
        default="0",
        help='gpus use for training',
    )

    parser.add_argument(
        '--max_files',
        type=int,
        default=None,
        help='number of expert files to read (leave as None unless doing ablations)',
    )
    parser.add_argument(
        '--max_experts',
        type=int,
        default=None,
        help='number of experts to read per file (leave as None unless doing ablations)',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=3407,
        help='set seed',
    )
    parser.add_argument(
        '--distilled_time',
        type=str,
        default=None,
        help='distilled time',
    )
    parser.add_argument(
        '--distill_loss',
        type=str,
        default=None,
        help='loss function',
    )
    parser.add_argument('--num_workers', type=int, default=0, help='num workers')
    parser.add_argument('--feature_epoch_no_mom', type=int, default=None, help='feature_epoch_no_mom')
    args = parser.parse_args()
    main(args)
