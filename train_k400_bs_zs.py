'''
bs:baseline module
zs:zero-shot traning
'''
import os
import sys
import time
import argparse
from timm import create_model
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.cuda.amp import GradScaler
import torchvision
import torch.optim as optim
from utils.utils import init_distributed_mode, AverageMeter, reduce_tensor, accuracy
from utils.logger import setup_logger
import clip

from pathlib import Path
import yaml
import pprint
from dotmap import DotMap
import numpy as np

import datetime
import shutil
from contextlib import suppress
from modules import cswin

from datasets.dataset import Video_dataset
# from modules.video_clip import video_header, VideoCLIP
from modules.no_temporal_video_clip import video_header,VideoCLIP
from utils.Augmentation import get_augmentation, randAugment
from utils.solver import _lr_scheduler
from modules.text_prompt import text_prompt
'''
对每个epoch训练模型进行保存
'''
def epoch_saving(epoch, model, optimizer, filename):
    torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, filename) #just change to your preferred folder/filename
'''
对目前最好的模型进行保存
'''
def best_saving(working_dir, epoch, model, optimizer):
    best_name = '{}/model_best.pt'.format(working_dir)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, best_name)  # just change to your preferred folder/filename
'''
去掉参数key中的module.
'''
def update_dict(dict):
    new_dict = {}
    for k, v in dict.items():
        new_dict[k.replace('module.', '')] = v
    return new_dict


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-cfg', type=str, default='clip.yaml', help='global config file')
    parser.add_argument('--log_time', default='no_temporal_clip')
    parser.add_argument('--dist_url', default='env://',help='url used to set up distributed training')
    parser.add_argument('--world_size', default=1, type=int,help='number of distributed processes')
    parser.add_argument("--local_rank", type=int,default=0,help='local rank for DistributedDataParallel')
    parser.add_argument(
        "--precision",
        choices=["amp", "fp16", "fp32"],
        default="amp",
        help="Floating point precition."
    )
    args = parser.parse_args()
    return args
'''
获取cswin模型最后norm层的维度
'''
def get_cswinmodel_pa(model):
    ks=[]
    vs=[]
    for k,v in model.named_parameters():
        ks.append(k)
        vs.append(v)
    print("The number of parameters of the cs_model is {}".format(len(ks)))
    return vs[-1].shape[0]

def main(args):
    global best_prec1
    """ Training Program """
    init_distributed_mode(args)
    if args.distributed:
        print('[INFO] turn on distributed train', flush=True)
    else:
        print('[INFO] turn off distributed train', flush=True)
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    working_dir = os.path.join('records/exp_full_shot', config['data']['dataset'], config['network']['arch'] ,args.log_time)
    if dist.get_rank() == 0:
        Path(working_dir).mkdir(parents=True, exist_ok=True)
        shutil.copy(args.config, working_dir)
        shutil.copy('train_k400_bs_zs.py', working_dir)

    # build logger, print env and config
    ## 运行过程中记录的打印文档
    logger = setup_logger(output=working_dir,
                          distributed_rank=dist.get_rank(),
                          name=f'CLIP_video_cls')
    logger.info("------------------------------------")
    logger.info("Environment Versions:")
    logger.info("- Python: {}".format(sys.version))
    logger.info("- PyTorch: {}".format(torch.__version__))
    logger.info("- TorchVison: {}".format(torchvision.__version__))
    logger.info("------------------------------------")
    pp = pprint.PrettyPrinter(indent=4)
    logger.info(pp.pformat(config))
    logger.info("------------------------------------")
    logger.info("storing name: {}".format(working_dir))



    config = DotMap(config)


    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        cudnn.benchmark = True
    '''
    训练与验证数据加载
    '''
    transform_train = get_augmentation(True, config)
    transform_val = get_augmentation(False, config)
    if config.data.randaug.N:
        transform_train = randAugment(transform_train, config)
    logger.info('train transforms: {}'.format(transform_train.transforms))
    logger.info('val transforms: {}'.format(transform_val.transforms))

    train_data = Video_dataset(
            config.data.train_root, config.data.train_list,
            config.data.label_list, num_segments=config.data.num_segments,
            modality=config.data.modality,
            image_tmpl=config.data.image_tmpl, random_shift=config.data.random_shift,
            transform=transform_train,new_length=config.data.seg_length)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    train_loader = DataLoader(train_data,
                              batch_size=config.data.batch_size, num_workers=config.data.workers,
                              sampler=train_sampler,drop_last=False)
    '''
    few-shot 设置
    每个类别选择shot条视频，此操作重复n_repeat次
    '''
#     if config.data.shot:
#         cls_dict = {}
#         for item  in train_data.video_list:
#             if item.label not in cls_dict:
#                 cls_dict[item.label] = [item]
#             else:
#                 cls_dict[item.label].append(item)
#         import random
#         select_vids = []
#         K = config.data.shot
#         for category, v in cls_dict.items():
#             slice = random.sample(v, K)
#             select_vids.extend(slice)
#         n_repeat = len(train_data.video_list) // len(select_vids)
#         train_data.video_list = select_vids * n_repeat
#     print(len(train_data))
    val_data = Video_dataset(
            config.data.val_root, config.data.val_list,
            config.data.label_list, num_segments=config.data.num_segments,
            modality=config.data.modality,
            image_tmpl=config.data.image_tmpl, random_shift=config.data.random_shift,
            transform=transform_train,new_length=config.data.seg_length)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)
    val_loader = DataLoader(val_data,
                              batch_size=config.data.batch_size, num_workers=config.data.workers,
                              sampler=val_sampler,drop_last=False)

    '''
    clip :text_encoder
    class feature：分类器
    '''
    clip_model, clip_state_dict = clip.load(config.network.arch,
                             device='cpu',jit=False,
                             internal_modeling=config.network.tm,
                             T=config.data.num_segments,
                             dropout=config.network.drop_out,
                            emb_dropout=config.network.emb_dropout,
                            pretrain=config.network.init,
                            joint_st=config.network.joint_st)

    classes, _, text_dict = text_prompt(train_data)# classes: 400*77
    n_class = text_dict[0].size(0)
    clip_model.eval()
    with torch.no_grad():
        classes_features = clip_model.encode_text(classes)
#     classes_features=torch.
    '''
    模型
    '''
#     video_head = video_header(config.network.sim_header,clip_state_dict)#sim_header='transfer'
#     cswin_model=create_model(config.network.cwin_arch, pretrained=False, )## cswin 模型构造
#     norm_dim=get_cswinmodel_pa(cswin_model)
#     cwsin_para=torch.load(config.network.cwin_pretrain,map_location='cpu')["state_dict_ema"]

#     cswin_model.load_state_dict(cwsin_para,strict=False)
    #model_full=Video_CSwin(cswin_model,video_head,config.data.num_segments,norm_dim)
    model_full = VideoCLIP(clip_model, config.data.num_segments)
    print('Model loading complete ')
    '''
    损失函数
    '''
    criterion = torch.nn.CrossEntropyLoss()

    '''
    优化策略
    '''
    cswin_params = []
    other_params = []
    for name, param in model_full.named_parameters():
        if 'visual' in name and 'control_point' not in name:
            cswin_params.append(param)
        elif 'logit_scale' in name:
            cswin_params.append(param)
        else:
            other_params.append(param)
    optimizer = optim.AdamW([{'params': cswin_params, 'lr': config.solver.lr * config.solver.clip_ratio},
                            {'params': other_params, 'lr': config.solver.lr}],
                            betas=(0.9, 0.98), lr=config.solver.lr, eps=1e-8,
                            weight_decay=config.solver.weight_decay)

    lr_scheduler = _lr_scheduler(config, optimizer)
    '''
    单机多卡分布
    '''
    if args.distributed:
        model_full = DistributedDataParallel(model_full.cuda(), device_ids=[args.gpu])
        model_without_ddp = model_full.module
    '''
    预测
    '''
    start_epoch = config.solver.start_epoch
    scaler = GradScaler() if args.precision == "amp" else None
    best_prec1 = 0.0
    ## config.solver.evaluate 设置参数
    if config.solver.evaluate:
        logger.info(("===========evaluate==========="))
        prec1 = validate(
            start_epoch,
            val_loader, device,
            model_full, config, classes_features, logger)
        return

    '''
    训练
    '''
    for epoch in range(start_epoch, config.solver.epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
        ## 训练一个epoch
        train(model_full, train_loader, optimizer, criterion, scaler,
              epoch, device, lr_scheduler, config, classes_features, logger)
        ## 每个epoch的模型都进行保存
        filename = "{}/{}_epoch{}_model.pt".format(working_dir,config.network.cwin_arch,epoch)
        epoch_saving(epoch, model_without_ddp, optimizer, filename)
        ## 验证一次
        if (epoch+1) % config.logging.eval_freq == 0:  # and epoch>0   config.logging.eval_freq =1
            prec1 = validate(epoch, val_loader, device, model_full, config, classes_features, logger)

            if dist.get_rank() == 0:
                is_best = prec1 > best_prec1
                best_prec1 = max(prec1, best_prec1)
                logger.info('Testing: {}/{}'.format(prec1,best_prec1))
                logger.info('Saving:')
                filename = "{}/{}_last_model.pt".format(working_dir,config.network.cwin_arch)

                epoch_saving(epoch, model_without_ddp, optimizer, filename)
                if is_best:
                    best_saving(working_dir, epoch, model_without_ddp, optimizer)
def train(model, train_loader, optimizer, criterion, scaler,
          epoch, device, lr_scheduler, config, text_embedding, logger):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    model.train()
    autocast = torch.cuda.amp.autocast if args.precision == 'amp' else suppress
    end=time.time()
    for i,(images, list_id) in enumerate(train_loader):
        if config.solver.type != 'monitor':
            if (i + 1) == 1 or (i + 1) % 10 == 0:
                lr_scheduler.step(epoch + i / len(train_loader))
        data_time.update(time.time() - end)
        # b t3 h w
        images = images.view((-1, config.data.num_segments*config.data.seg_length, 3) + images.size()[-2:])  # b t 3 h w
        b, t, c, h, w = images.size()## b :batch_size  t:sample_frames_num 3:RGB
#         images= images.view(-1, c, h, w)
        with autocast():
            images = images.to(device)
            logits = model(images, text_embedding) # B  400

            loss = criterion(logits, list_id.to(device))

            # loss regularization
            loss = loss / config.solver.grad_accumulation_steps
        if scaler is not None:
            # back propagation
            scaler.scale(loss).backward()

            if (i + 1) % config.solver.grad_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            loss.backward()
            if (i + 1) % config.solver.grad_accumulation_steps == 0:
                optimizer.step()  # update param
                optimizer.zero_grad()
        losses.update(loss.item(), logits.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        cur_iter = epoch * len(train_loader) +  i
        max_iter = config.solver.epochs * len(train_loader)
        eta_sec = batch_time.avg * (max_iter - cur_iter + 1)
        eta_sec = str(datetime.timedelta(seconds=int(eta_sec)))
        if i % config.logging.print_freq == 0:## config.logging.print_freq=10
            logger.info(('Epoch: [{0}][{1}/{2}], lr: {lr:.2e}, eta: {3}\t'
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                             epoch, i, len(train_loader), eta_sec, batch_time=batch_time, data_time=data_time, loss=losses,
                             lr=optimizer.param_groups[-1]['lr'])))
def validate(epoch, val_loader, device, model, config, text_embedding, logger):
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.eval()
    with torch.no_grad():
        for i, (images, class_id) in enumerate(val_loader):
            images = images.view((-1, config.data.num_segments*config.data.seg_length, 3) + images.size()[-2:])
            b, t, c, h, w = images.size()
            class_id = class_id.to(device)
            text_embedding = text_embedding.to(device)
            images = images.to(device)


            image_embedding = model.module.encode_image(images)
            image_embedding /= image_embedding.norm(dim=-1, keepdim=True)
            text_embedding /= text_embedding.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_embedding @ text_embedding.T)

            prec = accuracy(similarity, class_id, topk=(1, 5))
            prec1 = reduce_tensor(prec[0])
            prec5 = reduce_tensor(prec[1])

            top1.update(prec1.item(), class_id.size(0))
            top5.update(prec5.item(), class_id.size(0))

            if i % config.logging.print_freq == 0:
                logger.info(
                    ('Test: [{0}/{1}]\t'
                     'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                     'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                         i, len(val_loader), top1=top1, top5=top5)))

    logger.info(('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}' .format(top1=top1, top5=top5)))
    return top1.avg
if __name__ == '__main__':
    args = get_parser()
    main(args)

