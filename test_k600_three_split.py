'''
测试 k600、ucf101和hmdb51 三种划分
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
import pickle

import datetime
import shutil
from contextlib import suppress
from modules import cswin
from datasets.transforms import GroupScale, GroupCenterCrop, Stack, ToTorchFormatTensor, GroupNormalize, \
    GroupOverSample, GroupFullResSample
from datasets.dataset_k600_test import Video_dataset
from modules.video_clip_1layer_gain import video_header, VideoCLIP
# from modules.cswin_video_bs import video_header, Video_CSwin
from utils.Augmentation import get_augmentation, randAugment
from utils.solver import _lr_scheduler
from modules.text_prompt import text_prompt

'''
参数的传入
'''


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='global config file')
    parser.add_argument('--weights', type=str, default=None)
    parser.add_argument('--log_time', default='front_lay')  ## 6层时序层
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument("--local_rank", type=int,
                        help='local rank for DistributedDataParallel')
    parser.add_argument(
        "--precision",
        choices=["amp", "fp16", "fp32"],
        default="amp",
        help="Floating point precition."
    )
    parser.add_argument('--test_crops', type=int, default=3)
    parser.add_argument('--test_clips', type=int, default=3)
    parser.add_argument('--dense', default=False, action="store_true",
                        help='use multiple clips for test')
    args = parser.parse_args()
    return args


'''
去掉参数key中的module.
'''


def update_dict(dict):
    new_dict = {}
    for k, v in dict.items():
        new_dict[k.replace('module.', '')] = v
    return new_dict


'''
获取cswin模型最后norm层的维度
'''


def get_cswinmodel_pa(model):
    ks = []
    vs = []
    for k, v in model.named_parameters():
        ks.append(k)
        vs.append(v)
    print("The number of parameters of the cs_model is {}".format(len(ks)))
    return vs[-1].shape[0]


'''
主程序
'''


def main(args):
    init_distributed_mode(args)
    ## config
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = DotMap(config)
    ## device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        cudnn.benchmark = True
    '''
    训练保存文件
    '''
    save_root = os.path.join('test_records', config['data']['dataset'], config['network']['arch'], args.log_time)
    if dist.get_rank() == 0:
        Path(save_root).mkdir(parents=True, exist_ok=True)
    '''
    clip模型
    '''
    clip_model, clip_state_dict = clip.load(config.network.arch,
                                            device='cpu', jit=False,
                                            internal_modeling=config.network.tm,
                                            T=config.data.num_segments,
                                            dropout=config.network.drop_out,
                                            emb_dropout=config.network.emb_dropout,
                                            pretrain=config.network.init,
                                            joint_st=config.network.joint_st)
    if args.precision == "amp" or args.precision == "fp32":
        clip_model = clip_model.float()

    ## crop
    input_mean = [0.48145466, 0.4578275, 0.40821073]
    input_std = [0.26862954, 0.26130258, 0.27577711]

    # rescale size
    if 'something' in config.data.dataset:
        scale_size = (240, 320)
    else:
        scale_size = 256 if config.data.input_size == 224 else config.data.input_size

    # crop size
    input_size = config.data.input_size

    # control the spatial crop
    if args.test_crops == 1:  # one crop
        cropping = torchvision.transforms.Compose([
            GroupScale(scale_size),
            GroupCenterCrop(input_size),
        ])
    elif args.test_crops == 3:  # do not flip, so only 3 crops (left right center)
        cropping = torchvision.transforms.Compose([
            GroupFullResSample(
                crop_size=input_size,
                scale_size=scale_size,
                flip=False)
        ])
    elif args.test_crops == 5:  # do not flip, so only 5 crops
        cropping = torchvision.transforms.Compose([
            GroupOverSample(
                crop_size=input_size,
                scale_size=scale_size,
                flip=False)
        ])
    elif args.test_crops == 10:
        cropping = torchvision.transforms.Compose([
            GroupOverSample(
                crop_size=input_size,
                scale_size=scale_size,
            )
        ])
    else:
        raise ValueError("Only 1, 3, 5, 10 crops are supported while we got {}".format(args.test_crops))
    '''
    实验模型
    '''
    video_head = video_header(config.network.sim_header, clip_state_dict)
    # cswin_model = create_model(config.network.cwin_arch, pretrained=False, )
    # norm_dim = get_cswinmodel_pa(cswin_model)
    model_full = VideoCLIP(clip_model, video_head,config.data.num_segments)  # Video_CSwin(cswin_model, video_head, config.data.num_segments,norm_dim)
    '''
    参数加载
    args.weights：参数
    '''
    if os.path.isfile(args.weights):
        checkpoint = torch.load(args.weights, map_location='cpu')
        if dist.get_rank() == 0:
            print('load model: epoch {}'.format(checkpoint['epoch']))

        model_full.load_state_dict(update_dict(checkpoint['model_state_dict']))
        del checkpoint

    if args.distributed:
        model_full = DistributedDataParallel(model_full.cuda(), device_ids=[args.gpu], find_unused_parameters=True)
    res_be=torch.zeros((3))
    res_af=torch.zeros((3))
    ## val data
    val_list=[config.data.val_list1,config.data.val_list2,config.data.val_list3]
    label_list=[config.data.label_list1,config.data.label_list2,config.data.label_list3]
    for i in range(1,4):
        print('now split is {}'.format(i))
        val_list_now=val_list[i-1]
        label_list_now=label_list[i-1]
        print(val_list_now)
        print(label_list_now)
        val_data = Video_dataset(
            config.data.val_root, val_list_now ,label_list_now,
            random_shift=False, num_segments=config.data.num_segments,
            modality=config.data.modality,
            image_tmpl=config.data.image_tmpl,
            test_mode=True,
            transform=torchvision.transforms.Compose([
                cropping,
                Stack(roll=False),
                ToTorchFormatTensor(div=True),
                GroupNormalize(input_mean, input_std),
            ]),
            dense_sample=args.dense,
            test_clips=args.test_clips,
            new_length=config.data.seg_length)

        val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)
        val_loader = DataLoader(val_data,
                                batch_size=config.data.batch_size, num_workers=config.data.workers,
                                sampler=val_sampler, pin_memory=True, drop_last=False)

        ## clip生成的判别器，使用
        classes, _, text_dict = text_prompt(val_data)  # classes: 400*77
        n_class = text_dict[0].size(0)
        clip_model.eval()
        with torch.no_grad():
            classes_features = clip_model.encode_text(classes)


    #     print('model loading completed')
        '''
        预测
        '''

        prec1,front_pre1 = validate(
            val_loader, device,
            model_full, config, classes_features, args.test_crops, args.test_clips, save_root,i)
        res_af[i-1]=prec1
        res_be[i-1]=front_pre1


        # print(prec1, model_full.module.ratio)
        # return
    print('时序前为{}{}'.format(torch.mean(res_be),torch.std(res_be)))
    print('时序后为{}{}'.format(torch.mean(res_af),torch.std(res_af)))

def validate(val_loader, device, model, config, text_features, test_crops, test_clips, save_root,now_split):
    top1 = AverageMeter()
    top5 = AverageMeter()
    front_top1 = AverageMeter()
    front_top5 = AverageMeter()
    model.eval()
    proc_start_time = time.time()

    sim_logits = []  #
    labels = []  #
    i_features = []
    predics_top5 = []
    front_features = []

    with torch.no_grad():
        n_class = text_features.size(0)

        for i, (image, class_id) in enumerate(val_loader):
            #             print(image.shape)
            batch_size = class_id.numel()
            num_crop = test_crops

            num_crop *= test_clips  # 4 clips for testing when using dense sample

            class_id = class_id.to(device)
            text_features = text_features.to(device)
            #             n_seg = config.data.num_segments
            #             image = image.view((-1, n_seg, 3) + image.size()[-2:])
            image = image.view((-1, config.data.num_segments * config.data.seg_length, 3) + image.size()[-2:])
            b, t, c, h, w = image.size()
            #             print(b, t, c, h, w )
            # image_input = image.to(device).view(-1, c, h, w)
            image = image.to(device)
            image_features, front_image_emb = model.module.encode_image(image)
            #             print("image_features is {}".format(image_features.shape))
            cnt_time = time.time() - proc_start_time

            image_features = image_features.reshape(batch_size, num_crop, -1).mean(1)  # bs dim
            front_image_emb = front_image_emb.reshape(batch_size, num_crop, -1).mean(1)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            front_image_emb /= front_image_emb.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            similarity = (100.0 * image_features @ text_features.T)
            similarity = similarity.view(batch_size, -1, n_class).softmax(dim=-1)
            similarity = similarity.mean(dim=1, keepdim=False)  # bs 200

            front_similarity = (100.0 * front_image_emb @ text_features.T)
            front_similarity = front_similarity.view(batch_size, -1, n_class).softmax(dim=-1)
            front_similarity = front_similarity.mean(dim=1, keepdim=False)  # bs 200

            prec5 = similarity.topk(5)
            prec5 = prec5.indices

            ########## gathering
            i_features.append(concat_all_gather(image_features))
            front_features.append(concat_all_gather(front_image_emb))
            sim_logits.append(concat_all_gather(similarity))
            labels.append(concat_all_gather(class_id))
            predics_top5.append(concat_all_gather(prec5))
            ##########

            prec = accuracy(similarity, class_id, topk=(1, 5))
            prec1 = reduce_tensor(prec[0])
            prec5 = reduce_tensor(prec[1])
            top1.update(prec1.item(), class_id.size(0))
            top5.update(prec5.item(), class_id.size(0))

            front_prec = accuracy(front_similarity, class_id, topk=(1, 5))
            front_prec1 = reduce_tensor(front_prec[0])
            front_prec5 = reduce_tensor(front_prec[1])
            front_top1.update(front_prec1.item(), class_id.size(0))
            front_top5.update(front_prec5.item(), class_id.size(0))

            if i % config.logging.print_freq == 0 and dist.get_rank() == 0:
                runtime = float(cnt_time) / (i + 1) / (batch_size * dist.get_world_size())
                print(
                    ('Test: [{0}/{1}], average {runtime:.4f} sec/video \t'
                     'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                     'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                     'front_Prec@1 {front_top1.val:.3f} ({front_top1.avg:.3f})\t'
                     'front_Prec@5 {front_top5.val:.3f} ({front_top5.avg:.3f})\t'.format(
                        i, len(val_loader), runtime=runtime, top1=top1, top5=top5, front_top1=front_top1,
                        front_top5=front_top5)))

    if dist.get_rank() == 0:
        print('-----Full-classes Evaluation------')
        print('Overall Top1 {:.03f}% Top5 {:.03f}%'.format(top1.avg, top5.avg))

        ## half-classes evaluation
        sim, la = sim_logits[0], labels[0]
        vid_feat = i_features[0]
        pre_lab5 = predics_top5[0]
        front_vid_feat = front_features[0]
        for i in range(1, len(sim_logits)):
            sim = torch.cat((sim, sim_logits[i]), 0)
            la = torch.cat((la, labels[i]), 0)
            vid_feat = torch.cat((vid_feat, i_features[i]), 0)
            pre_lab5 = torch.cat((pre_lab5, predics_top5[i]), 0)
            front_vid_feat = torch.cat((front_vid_feat, front_features[i]), 0)
        las = torch.unsqueeze(la, 1)
        #         print(pre_lab5.shape,la.shape)
        labels = torch.cat((pre_lab5.cpu(), las.cpu()), 1)
        save_pickle(labels, vid_feat.cpu(), front_vid_feat.cpu(), text_features.cpu(), save_root,now_split)

        '''
        half-classes 分类
        '''
        # acc_split, acc_split_top5 = multi_split_test(vid_feat.cpu(), text_features.cpu(), la.cpu())
        # accuracy_split, accuracy_split_std = np.mean(acc_split), np.std(acc_split)
        # accuracy_split_top5, accuracy_split_top5_std = np.mean(acc_split_top5), np.std(acc_split_top5)
        #
        # front_acc_split, front_acc_split_top5 = multi_split_test(front_vid_feat.cpu(), text_features.cpu(), la.cpu())
        # front_accuracy_split, front_accuracy_split_std = np.mean(front_acc_split), np.std(front_acc_split)
        # front_accuracy_split_top5, front_accuracy_split_top5_std = np.mean(front_acc_split_top5), np.std(
        #     front_acc_split_top5)
        # print('-----Half-classes Evaluation after layer-----')
        # print('Top1: mean {:.03f}%, std {:.03f}%'.format(accuracy_split, accuracy_split_std))
        # print('Top5: mean {:.03f}%, std {:.03f}%'.format(accuracy_split_top5, accuracy_split_top5_std))
        #
        # print('-----Half-classes Evaluation before layer-----')
        # print('Top1: mean {:.03f}%, std {:.03f}%'.format(front_accuracy_split, front_accuracy_split_std))
        # print('Top5: mean {:.03f}%, std {:.03f}%'.format(front_accuracy_split_top5, front_accuracy_split_top5_std))

    return top1.avg,front_top1.avg


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output.cpu()


def save_pickle(labels, visul_video, front_vid_feat, text_features, save_root,now_split):
    flabel = open(os.path.join(save_root, 'pre_labels{}'.format(str(now_split))), 'wb')
    pickle.dump(labels, flabel)
    flabel.close()
    ffeas = open(os.path.join(save_root, 'video_features{}'.format(str(now_split))), 'wb')
    pickle.dump(visul_video, ffeas)
    ffeas.close()
    ffeas = open(os.path.join(save_root, 'before_video_features{}').format(str(now_split)), 'wb')
    pickle.dump(front_vid_feat, ffeas)
    ffeas.close()
    ffeas = open(os.path.join(save_root, 'cls_feature{}').format(str(now_split)), 'wb')
    pickle.dump(text_features, ffeas)
    ffeas.close()


def compute_accuracy(vis_emb, text_emb, label):
    n_class = len(text_emb)
    n_samples = len(vis_emb)
    similarity = (100.0 * vis_emb @ text_emb.T)
    similarity = similarity.view(n_samples, -1, n_class).softmax(dim=-1)
    similarity = similarity.mean(dim=1, keepdim=False)  # b 101
    prec = accuracy(similarity, label, topk=(1, 5))
    return prec[0], prec[1]


## 数据集的10次随机划分 选一半进行测试，最重要操作是。标签的重新映射
def multi_split_test(vis_embs, text_embs, true_label):
    # vis_embs: [10000, 768]
    # text_embs: [101, 768]
    # true_label: [10000,]
    full_acc1, full_acc5 = compute_accuracy(vis_embs, text_embs, true_label)

    # Calculate accuracy per split
    # Only when the model has been trained on a different dataset
    true_label = true_label.numpy()
    accuracy_split, accuracy_split_top5 = np.zeros(10), np.zeros(10)
    for split in range(len(accuracy_split)):
        np.random.seed(split)
        sel_classes = np.random.permutation(len(text_embs))[:len(text_embs) // 2]  # 一半的类别
        sel = [l in sel_classes for l in true_label]  # 判断每一条视频的标签是否在测试的类别内
        subclasses = np.unique(true_label[sel])  ## 测试的类别标签
        #### 标签映射
        tl = np.array([int(np.where(l == subclasses)[0]) for l in true_label[sel]])  ## true_label[sel] 视频数
        tl = torch.from_numpy(tl)  ## 原始相对所有类别的标签--》相对于一半类别的标签
        acc, acc5 = compute_accuracy(vis_embs[sel], text_embs[subclasses], tl)
        accuracy_split[split] = acc
        accuracy_split_top5[split] = acc5

    return accuracy_split, accuracy_split_top5


if __name__ == '__main__':
    args = get_parser()
    main(args)

