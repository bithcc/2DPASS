#!/usr/bin/env python
# encoding: utf-8
'''
@author: Xu Yan
@file: main.py
@time: 2021/12/7 22:21
'''

import os#python与操作系统进行交互的接口
import yaml#配置文件相关选项
import torch#torch库
import datetime#日期时间相关库
import importlib#importlib是python的一个库，通过导入importlib，调用import_module()方法，传入用户想要获取的模块对应的路径字符串，即可获取一个模块module，module可以调用这个test模块下的所有属性和方法
import numpy as np#python的数值计算扩展
import pytorch_lightning as pl

from easydict import EasyDict#通过easydict可以用简单的方法访问字典元素，设置属性等
from argparse import ArgumentParser#读取命令行参数
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.profiler import SimpleProfiler
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from dataloader.dataset import get_model_class, get_collate_class
from dataloader.pc_dataset import get_pc_model_class
from pytorch_lightning.callbacks import LearningRateMonitor

import warnings
warnings.filterwarnings("ignore")

#这一部分从yaml中读取参数文件，主要是一些模型的超参数和数据集读取的超参数
def load_yaml(file_name):
    with open(file_name, 'r') as f:
        try:
            config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            config = yaml.load(f)
    return config

#这一部分规定了对于命令行参数的解析方法
def parse_config():
    parser = ArgumentParser()
    # general
    parser.add_argument('--gpu', type=int, nargs='+', default=(0,), help='specify gpu devices')#选择gpu
    parser.add_argument("--seed", default=0, type=int)#设置随机种子
    parser.add_argument('--config_path', default='config/2DPASS-semantickitti.yaml')#读取参数文件
    # training
    parser.add_argument('--log_dir', type=str, default='default', help='log location')#日志存放位置
    parser.add_argument('--monitor', type=str, default='val/mIoU', help='the maximum metric')#精度评测方法，使用mIoU
    parser.add_argument('--stop_patience', type=int, default=50, help='patience for stop training')#多少轮精度不变化停止训练
    parser.add_argument('--save_top_k', type=int, default=1, help='save top k checkpoints, use -1 to checkpoint every epoch')#保存精度最好的几个模型
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1, help='check_val_every_n_epoch')#多少轮训练验证一次
    parser.add_argument('--SWA', action='store_true', default=False, help='StochasticWeightAveraging')#是否使用随机梯度平均
    parser.add_argument('--baseline_only', action='store_true', default=False, help='training without 2D')#baseline方法，不使用2d数据训练
    # testing
    parser.add_argument('--test', action='store_true', default=False, help='test mode')#测试模式
    parser.add_argument('--fine_tune', action='store_true', default=False, help='fine tune mode')#微调模式
    parser.add_argument('--pretrain2d', action='store_true', default=False, help='use pre-trained 2d network')#使用预训练的2d网络
    parser.add_argument('--num_vote', type=int, default=1, help='number of voting in the test')#测试时的vote数目
    parser.add_argument('--submit_to_server', action='store_true', default=False, help='submit on benchmark')#提交到服务器
    parser.add_argument('--checkpoint', type=str, default=None, help='load checkpoint')#加载预训练权重
    # debug
    parser.add_argument('--debug', default=False, action='store_true')#debug模式

    args = parser.parse_args()#从命令行读取参数数据
    config = load_yaml(args.config_path)#从yaml文件中读取参数数据
    config.update(vars(args))  # override the configuration using the value in args 使用命令行中读取的参数覆盖从yaml文件中读取的参数

    # voting test
    if args.test:
        config['dataset_params']['val_data_loader']['batch_size'] = args.num_vote#是否使用投票法
    if args.num_vote > 1:
        config['dataset_params']['val_data_loader']['rotate_aug'] = True#如果使用投票法，对一个数据进行旋转增强
        config['dataset_params']['val_data_loader']['transform_aug'] = True#如果使用投票法，对一个数据进行transform增强
    if args.debug:
        config['dataset_params']['val_data_loader']['batch_size'] = 2
        config['dataset_params']['val_data_loader']['num_workers'] = 0

    return EasyDict(config)#将读取的参数使用存储为easy_dict模式


def build_loader(config):#如何从config中建立dataloader
    pc_dataset = get_pc_model_class(config['dataset_params']['pc_dataset_type'])#读取config里写明的点云数据类型，如semantickitti
    dataset_type = get_model_class(config['dataset_params']['dataset_type'])#读取数据格式，如voxel_dataset
    train_config = config['dataset_params']['train_data_loader']#读取训练数据参数，如num_worker，是否数据增强
    val_config = config['dataset_params']['val_data_loader']#读取验证数据参数
    train_dataset_loader, val_dataset_loader, test_dataset_loader = None, None, None

    if not config['test']:#train
        train_pt_dataset = pc_dataset(config, data_path=train_config['data_path'], imageset='train')#加载训练集数据
        val_pt_dataset = pc_dataset(config, data_path=val_config['data_path'], imageset='val')#加载验证集数据
        train_dataset_loader = torch.utils.data.DataLoader(
            dataset=dataset_type(train_pt_dataset, config, train_config),
            batch_size=train_config["batch_size"],
            collate_fn=get_collate_class(config['dataset_params']['collate_type']),#以voxel的方式读取原始点云数据
            shuffle=train_config["shuffle"],
            num_workers=train_config["num_workers"],
            pin_memory=True,
            drop_last=True
        )
        # config['dataset_params']['training_size'] = len(train_dataset_loader) * len(configs.gpu)
        val_dataset_loader = torch.utils.data.DataLoader(
            dataset=dataset_type(val_pt_dataset, config, val_config, num_vote=1),
            batch_size=val_config["batch_size"],#这里的num_vote和batch_size的区别是什么？
            collate_fn=get_collate_class(config['dataset_params']['collate_type']),
            shuffle=val_config["shuffle"],
            pin_memory=True,
            num_workers=val_config["num_workers"]
        )
    else:#test
        if config['submit_to_server']:#上传到服务器test
            test_pt_dataset = pc_dataset(config, data_path=val_config['data_path'], imageset='test', num_vote=val_config["batch_size"])
            test_dataset_loader = torch.utils.data.DataLoader(
                dataset=dataset_type(test_pt_dataset, config, val_config, num_vote=val_config["batch_size"]),
                batch_size=val_config["batch_size"],
                collate_fn=get_collate_class(config['dataset_params']['collate_type']),
                shuffle=val_config["shuffle"],
                num_workers=val_config["num_workers"]
            )
        else:#本地val
            val_pt_dataset = pc_dataset(config, data_path=val_config['data_path'], imageset='val', num_vote=val_config["batch_size"])
            val_dataset_loader = torch.utils.data.DataLoader(
                dataset=dataset_type(val_pt_dataset, config, val_config, num_vote=val_config["batch_size"]),
                batch_size=val_config["batch_size"],
                collate_fn=get_collate_class(config['dataset_params']['collate_type']),
                shuffle=val_config["shuffle"],
                num_workers=val_config["num_workers"]
            )

    return train_dataset_loader, val_dataset_loader, test_dataset_loader


if __name__ == '__main__':#直接执行当前文件，下面的函数才会被执行，否则在其他文件中调用当前文件，只会调用前面定义的几个函数
    # parameters
    configs = parse_config()
    print(configs)#运行当前函数时，顺带着从命令行中读取相应的命令行参数

    # setting
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, configs.gpu))
    #等价于os.environ["CUDA_VISIBLE_DEVICES"]='configs.gpu'也即使用参数中指定的gpu作为训练gpu，默认为0
    num_gpu = len(configs.gpu)

    # output path
    log_folder = 'logs/' + configs['dataset_params']['pc_dataset_type']#日志存储地址，这里是做了一个定义
    tb_logger = pl_loggers.TensorBoardLogger(log_folder, name=configs.log_dir, default_hp_metric=False)#调用tensorboard记录训练过程
    os.makedirs(f'{log_folder}/{configs.log_dir}', exist_ok=True)#创建存放地址
    profiler = SimpleProfiler(output_filename=f'{log_folder}/{configs.log_dir}/profiler.txt')#存储每一步的计算时间，以秒为单位
    np.set_printoptions(precision=4, suppress=True)
    #np.set_printoptions()用于控制python中小数的显示精度
    #np.set_printoptions(precision=None,threshold=None,linewidth=None,suppress=None,formatter=None)
    #  precision控制输出结果的精度，默认值为8。 threshold当数组元素总数过大时，设置显示的数字位数，其余用省略号代替（当数组元素总数大于设置值，控制输出值的个数为6个，当数组元素小于或者等于设置值的时候，全部显示。）当设置值为sys.maxsize(需要导入sys库)，则会输出所有元素。
    #  linewidth，每行字符的数目，其余的数值会换到下一行，suppress，小数是否需要以科学计数法的形式输出，formatter，自定义输出规则

    # save the backup files  #保存训练的配置文件
    backup_dir = os.path.join(log_folder, configs.log_dir, 'backup_files_%s' % str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')))
    if not configs['test']:#train的时候才保存
        os.makedirs(backup_dir, exist_ok=True)
        os.system('cp main.py {}'.format(backup_dir))
        os.system('cp dataloader/dataset.py {}'.format(backup_dir))
        os.system('cp dataloader/pc_dataset.py {}'.format(backup_dir))
        os.system('cp {} {}'.format(configs.config_path, backup_dir))
        os.system('cp network/base_model.py {}'.format(backup_dir))
        os.system('cp network/baseline.py {}'.format(backup_dir))
        os.system('cp {}.py {}'.format('network/' + configs['model_params']['model_architecture'], backup_dir))

    # reproducibility
    torch.manual_seed(configs.seed)#为了实验结果可以复现，设置随机种子
    torch.backends.cudnn.deterministic = True#每次返回的卷积算法将是确定的，配合上设置torch的随机种子为固定值的话，应该可以保证每次运行网络的时候相同输入的输出是固定的
    torch.backends.cudnn.benchmark = True#大部分情况下，设置这个flag可以让内置的cuDNN的auto-tuner自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题
    np.random.seed(configs.seed)#用于生成指定的随机数，当seed()中的参数被设置了之后，np.random.seed()可以按顺序产生一组固定的数组，如果使用相同的seed()值，则每次生成的随机数都相同。如果不设置这个值，那么每次生成的随机数不同
    config_path = configs.config_path

    train_dataset_loader, val_dataset_loader, test_dataset_loader = build_loader(configs)#从config文件中，构建data_loader
    model_file = importlib.import_module('network.' + configs['model_params']['model_architecture'])#这个地方指定了spvcnn模型
    my_model = model_file.get_model(configs)

    pl.seed_everything(configs.seed)#设置全局的随机种子
    checkpoint_callback = ModelCheckpoint(
        monitor=configs.monitor,
        mode='max',
        save_last=True,
        save_top_k=configs.save_top_k)#每一轮训练完成后保存checkpoint，保存的标准是最大的miou，保存最后一个，top_k是保存最好的几个

    if configs.checkpoint is not None:#如果设置了预加载的模型checkpoint，则加载此前的checkpoint
        print('load pre-trained model...')
        if configs.fine_tune or configs.test or configs.pretrain2d:#微调、测试、预训练。这个时候会使用新的config进行调参
            my_model = my_model.load_from_checkpoint(configs.checkpoint, config=configs, strict=(not configs.pretrain2d))
        else:
            # continue last training #接续训练，这个时候会延续训练时的config进行接续训练
            my_model = my_model.load_from_checkpoint(configs.checkpoint)

    if configs.SWA:
        swa = [StochasticWeightAveraging(swa_epoch_start=configs.train_params.swa_epoch_start, annealing_epochs=1)]#swa随机梯度下降，设置swa开始的轮次和过渡轮次，但是我看config文件里并没有相应的参数设置
    else:
        swa = []

    if not configs.test:
        # init trainer
        print('Start training...')
        trainer = pl.Trainer(gpus=[i for i in range(num_gpu)],#多卡分布式训练
                             accelerator='ddp',
                             max_epochs=configs['train_params']['max_num_epochs'],
                             resume_from_checkpoint=configs.checkpoint if not configs.fine_tune and not configs.pretrain2d else None,
                             callbacks=[checkpoint_callback,
                                        LearningRateMonitor(logging_interval='step'),#自动监控保存训练过程中每一步的学习率
                                        EarlyStopping(monitor=configs.monitor,
                                                      patience=configs.stop_patience,
                                                      mode='max',
                                                      verbose=True),
                                        ] + swa,#这一部分设置的是回调函数，也就是每一轮训练开始之前和完成之后要做的事情
                             logger=tb_logger,#tensorboard记录训练过程
                             profiler=profiler,#存储训练时间
                             check_val_every_n_epoch=configs.check_val_every_n_epoch,
                             gradient_clip_val=1,#用于控制梯度的裁剪（clipping）。梯度裁剪是一种优化技术，用于防止梯度爆炸(gradient explosion)和梯度消失(gradient vanishIng).
                             #gradient_clip_val参数的值表示要将梯度裁剪到的最大范数值。如果梯度的范数超过这个值，就会对梯度进行裁剪，将其缩小到指定的范围内
                             accumulate_grad_batches=1#计算几个batch的梯度平均进行更新，这一般是为了在小显卡上用minibatch训练大模型
                             )
        trainer.fit(my_model, train_dataset_loader, val_dataset_loader)

    else:
        print('Start testing...')
        assert num_gpu == 1, 'only support single GPU testing!'
        trainer = pl.Trainer(gpus=[i for i in range(num_gpu)],
                             accelerator='ddp',
                             resume_from_checkpoint=configs.checkpoint,
                             logger=tb_logger,
                             profiler=profiler)
        trainer.test(my_model, test_dataset_loader if configs.submit_to_server else val_dataset_loader)