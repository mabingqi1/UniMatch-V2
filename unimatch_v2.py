import argparse
from copy import deepcopy
import logging
import os
import pprint
import random
import numpy as np
import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset  
from torch.utils.tensorboard import SummaryWriter
import yaml
import SimpleITK as sitk


from dataset.semi import SemiDataset, SemiYHDataset, LABEL_PROJ_DICT
from model.semseg.dpt import DPT
from supervised import evaluate
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, init_log, AverageMeter
from util.dist_helper import setup_distributed


parser = argparse.ArgumentParser(description='UniMatch V2: Pushing the Limit of Semi-Supervised Semantic Segmentation')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--train_label_json', type=str, required=True)
parser.add_argument('--train_unlabel_json', type=str, required=True)
parser.add_argument('--val_json', type=str, required=True)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--local_rank', '--local-rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)
parser.add_argument('--infer', default=None, help='Test mode, no training will be performed')


def main():
    args = parser.parse_args()
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    rank, world_size = setup_distributed(port=args.port)

    if args.infer is not None:
    # 推理模式
        logger.info('Running in inference mode')

        model_configs = {
            'small': {'encoder_size': 'small', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'base': {'encoder_size': 'base', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'large': {'encoder_size': 'large', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'giant': {'encoder_size': 'giant', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        # 加载最佳模型
        model = DPT(**{**model_configs[cfg['backbone'].split('_')[-1]], 'nclass': cfg['nclass'], 'patch_size': cfg['patch_size']})
        checkpoint = torch.load(os.path.join(args.save_path, 'best.pth'), map_location='cpu')
        state_dict = checkpoint['model']
        for key in list(state_dict.keys()):
            state_dict[key.replace("module.", "")] = state_dict[key]
            state_dict.pop(key)
        model.load_state_dict(checkpoint['model'])
        model.cuda()
        model.eval()

        testset = SemiYHDataset(
            args.val_json, mode='val'
        )
        # testsampler = torch.utils.data.distributed.DistributedSampler(testset)
        testloader = DataLoader(
            testset, batch_size=1, pin_memory=False, num_workers=0, drop_last=False, 
            # sampler=testsampler
        )
        os.makedirs(os.path.join(args.save_path, 'outputsss'), exist_ok=True)
        # 推理并输出logits
        all_logits = []
        with torch.no_grad():
            for i, (img, mask) in enumerate(testloader):
                img = img.cuda()
                logits = model(img)
                all_logits.append(logits.cpu())
                pred = torch.argmax(logits, dim=1).cpu().numpy().astype(np.uint8)  # [B, H, W]
                pred = pred[0]  # [H, W]

                # 保存预测结果
                sitk_img = sitk.GetImageFromArray(pred)
                output_path = os.path.join(args.save_path, 'outputsss', f'pred_{i:04d}.nii.gz')
                sitk.WriteImage(sitk_img, output_path)
                # sitk.WriteImage(sitk.GetImageFromArray(mask[0].astype(np.uint8)), output_path.replace('pred', 'mask'))
                # raise
                print('successfully saved prediction to {}'.format(output_path))

        logger.info('INFER Finished!')
    
        return


    if rank == 0:
        all_args = {**cfg, **vars(args), 'ngpus': world_size}
        logger.info('{}\n'.format(pprint.pformat(all_args)))
        
        writer = SummaryWriter(args.save_path)
        
        os.makedirs(args.save_path, exist_ok=True)

    cudnn.enabled = True
    cudnn.benchmark = True

    ### MODEL ###
    model_configs = {
        'small': {'encoder_size': 'small', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'base': {'encoder_size': 'base', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'large': {'encoder_size': 'large', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'giant': {'encoder_size': 'giant', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    model = DPT(**{**model_configs[cfg['backbone'].split('_')[-1]], 'nclass': cfg['nclass'], 'patch_size': cfg['patch_size']})
    
    state_dict = torch.load(cfg['pretrain_weight'], map_location='cpu')
    s_or_t = next(iter(state_dict))
    keys_to_pop = []
    keys_to_rename = []
    for key in list(state_dict[s_or_t].keys()):
        if "dino_head" in key:
            keys_to_pop.append(key)
        if "ibot_head" in key:
            keys_to_pop.append(key)
        if "backbone." in key:
            keys_to_rename.append((key, key.replace("backbone.", "").replace("blocks.0.", "blocks.")))

    for key in keys_to_pop:
        state_dict[s_or_t].pop(key)
    for old_key, new_key in keys_to_rename:
        state_dict[s_or_t][new_key] = state_dict[s_or_t][old_key]
        state_dict[s_or_t].pop(old_key)

    fix_state_dict = state_dict[s_or_t]
    model.backbone.load_state_dict(fix_state_dict, strict=True)
        
    if cfg['lock_backbone']:
        model.lock_backbone()
    
    optimizer = AdamW(
        [
            {'params': [p for p in model.backbone.parameters() if p.requires_grad], 'lr': cfg['lr']},
            {'params': [param for name, param in model.named_parameters() if 'backbone' not in name], 'lr': cfg['lr'] * cfg['lr_multi']}
        ], 
        lr=cfg['lr'], betas=(0.9, 0.999), weight_decay=0.01
    )
    
    if rank == 0:
        print('MODEL | Total params: {:.1f}M'.format(count_params(model)))
        print('MODEL | Encoder params: {:.1f}M'.format(count_params(model.backbone)))
        print('MODEL | Decoder params: {:.1f}M\n'.format(count_params(model.head)))
    
    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], broadcast_buffers=False, output_device=local_rank, find_unused_parameters=True
    )
    
    model_ema = deepcopy(model)
    model_ema.eval()
    for param in model_ema.parameters():
        param.requires_grad = False
    
    ### DATASET ###
    trainset_u = SemiYHDataset(
        args.train_unlabel_json, mode='train_u', size=cfg['crop_size']
    )
    trainset_u = Subset(trainset_u, random.sample(range(len(trainset_u)), cfg['unlabel_num']))
    trainset_l = SemiYHDataset(
        args.train_label_json, mode='train_l', size=cfg['crop_size'], nsample=len(trainset_u)
    )
    valset = SemiYHDataset(
        args.val_json, mode='val'
    )
    # valset = Subset(valset, random.sample(range(len(valset)), 10))
    if rank == 0:
        print(f"DATASET | Train labeled set size: {len(trainset_l)}\n"
              f"DATASET | Train unlabeled set size: {len(trainset_u)}\n"
              f"DATASET | Val set size: {len(valset)}\n"
              )

    trainsampler_l = torch.utils.data.distributed.DistributedSampler(trainset_l)
    trainloader_l = DataLoader(
        trainset_l, batch_size=cfg['batch_size'], pin_memory=True, num_workers=8, drop_last=True, sampler=trainsampler_l
    )
    
    trainsampler_u = torch.utils.data.distributed.DistributedSampler(trainset_u)
    trainloader_u = DataLoader(
        trainset_u, batch_size=cfg['batch_size'], pin_memory=True, num_workers=8, drop_last=True, sampler=trainsampler_u
    )
    
    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(
        valset, batch_size=1, pin_memory=True, num_workers=0, drop_last=False, sampler=valsampler
    )

    ### CRITERION ###
    if cfg['criterion']['name'] == 'CELoss':
        criterion_l = nn.CrossEntropyLoss(**cfg['criterion']['kwargs']).cuda(local_rank)
    elif cfg['criterion']['name'] == 'OHEM':
        criterion_l = ProbOhemCrossEntropy2d(**cfg['criterion']['kwargs']).cuda(local_rank)
    else:
        raise NotImplementedError('%s criterion is not implemented' % cfg['criterion']['name'])

    criterion_u = nn.CrossEntropyLoss(reduction='none').cuda(local_rank)

    ### TRAIN ###
    total_iters = len(trainloader_u) * cfg['epochs']
    previous_best, previous_best_ema = 0.0, 0.0
    best_epoch, best_epoch_ema = 0, 0
    epoch = -1
    
    if os.path.exists(os.path.join(args.save_path, 'latest.pth')):
        checkpoint = torch.load(os.path.join(args.save_path, 'latest.pth'), map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        model_ema.load_state_dict(checkpoint['model_ema'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        previous_best = checkpoint['previous_best']
        previous_best_ema = checkpoint['previous_best_ema']
        best_epoch = checkpoint['best_epoch']
        best_epoch_ema = checkpoint['best_epoch_ema']
        
        if rank == 0:
            logger.info('TRAIN | Load from checkpoint at epoch %i\n' % epoch)

    for epoch in range(epoch + 1, cfg['epochs']):
        if rank == 0:
            logger.info('TRAIN | Epoch {:} | Previous Best: {:.2f} @ Epoch-{:} | '
                        'EMA Best: {:.2f} @ Epoch-{:}'.format(epoch, previous_best, best_epoch, previous_best_ema, best_epoch_ema))
        
        total_loss  = AverageMeter()
        total_loss_x = AverageMeter()
        total_loss_s = AverageMeter()
        total_mask_ratio = AverageMeter()

        trainloader_l.sampler.set_epoch(epoch)
        trainloader_u.sampler.set_epoch(epoch)

        loader = zip(trainloader_l, trainloader_u)
        
        model.train()

        for i, ((img_x, mask_x),
                (img_u_w, img_u_s1, img_u_s2, ignore_mask, cutmix_box1, cutmix_box2)) in enumerate(loader):
            img_x, mask_x = img_x.cuda(), mask_x.cuda()
            img_u_w, img_u_s1, img_u_s2 = img_u_w.cuda(), img_u_s1.cuda(), img_u_s2.cuda()
            ignore_mask, cutmix_box1, cutmix_box2 = ignore_mask.cuda(), cutmix_box1.cuda(), cutmix_box2.cuda()
            
            with torch.no_grad():
                pred_u_w = model_ema(img_u_w).detach()
                conf_u_w = pred_u_w.softmax(dim=1).max(dim=1)[0]
                mask_u_w = pred_u_w.argmax(dim=1)
            
            img_u_s1[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1] = img_u_s1.flip(0)[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1]
            img_u_s2[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1] = img_u_s2.flip(0)[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1]
            
            pred_x = model(img_x)
            pred_u_s1, pred_u_s2 = model(torch.cat((img_u_s1, img_u_s2)), comp_drop=True).chunk(2)
            
            mask_u_w_cutmixed1, conf_u_w_cutmixed1, ignore_mask_cutmixed1 = mask_u_w.clone(), conf_u_w.clone(), ignore_mask.clone()
            mask_u_w_cutmixed2, conf_u_w_cutmixed2, ignore_mask_cutmixed2 = mask_u_w.clone(), conf_u_w.clone(), ignore_mask.clone()

            mask_u_w_cutmixed1[cutmix_box1 == 1] = mask_u_w.flip(0)[cutmix_box1 == 1]
            conf_u_w_cutmixed1[cutmix_box1 == 1] = conf_u_w.flip(0)[cutmix_box1 == 1]
            ignore_mask_cutmixed1[cutmix_box1 == 1] = ignore_mask.flip(0)[cutmix_box1 == 1]
            
            mask_u_w_cutmixed2[cutmix_box2 == 1] = mask_u_w.flip(0)[cutmix_box2 == 1]
            conf_u_w_cutmixed2[cutmix_box2 == 1] = conf_u_w.flip(0)[cutmix_box2 == 1]
            ignore_mask_cutmixed2[cutmix_box2 == 1] = ignore_mask.flip(0)[cutmix_box2 == 1]

            loss_x = criterion_l(pred_x, mask_x.squeeze(1))

            loss_u_s1 = criterion_u(pred_u_s1, mask_u_w_cutmixed1)
            loss_u_s1 = loss_u_s1 * ((conf_u_w_cutmixed1 >= cfg['conf_thresh']) & (ignore_mask_cutmixed1 != 255))
            loss_u_s1 = loss_u_s1.sum() / (ignore_mask_cutmixed1 != 255).sum().item()

            loss_u_s2 = criterion_u(pred_u_s2, mask_u_w_cutmixed2)
            loss_u_s2 = loss_u_s2 * ((conf_u_w_cutmixed2 >= cfg['conf_thresh']) & (ignore_mask_cutmixed2 != 255))
            loss_u_s2 = loss_u_s2.sum() / (ignore_mask_cutmixed2 != 255).sum().item()
            
            loss_u_s = (loss_u_s1 + loss_u_s2) / 2.0
            
            loss = (loss_x + loss_u_s) / 2.0
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss.update(loss.item())
            total_loss_x.update(loss_x.item())
            total_loss_s.update(loss_u_s.item())

            mask_ratio = ((conf_u_w >= cfg['conf_thresh']) & (ignore_mask != 255)).sum().item() / (ignore_mask != 255).sum()
            total_mask_ratio.update(mask_ratio.item())

            iters = epoch * len(trainloader_u) + i
            lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * cfg['lr_multi']
            
            ema_ratio = min(1 - 1 / (iters + 1), cfg['ema_ratio_min'])
            
            for param, param_ema in zip(model.parameters(), model_ema.parameters()):
                param_ema.copy_(param_ema * ema_ratio + param.detach() * (1 - ema_ratio))
            for buffer, buffer_ema in zip(model.buffers(), model_ema.buffers()):
                buffer_ema.copy_(buffer_ema * ema_ratio + buffer.detach() * (1 - ema_ratio))
            
            if rank == 0:
                writer.add_scalar('train/loss_all', loss.item(), iters)
                writer.add_scalar('train/loss_x', loss_x.item(), iters)
                writer.add_scalar('train/loss_s', loss_u_s.item(), iters)
                writer.add_scalar('train/mask_ratio', mask_ratio, iters)

            if (i % 10 == 0) and (rank == 0):
                logger.info('TRAIN | Epoch {} | Iters: {:} | LR: {:.7f} | Total loss: {:.4f} | Loss x: {:.4f} | Loss s: {:.4f}, Mask ratio: '
                            '{:.3f}'.format(epoch, i, optimizer.param_groups[0]['lr'], total_loss.avg, total_loss_x.avg, 
                                            total_loss_s.avg, total_mask_ratio.avg))
        
        eval_mode = cfg['eval_mode']
        logger.info('Evaluation MAIN model...') if rank == 0 else None
        mDICE, dice_class = evaluate(model, valloader, eval_mode, cfg, rank, multiplier=cfg['patch_size'])
        logger.info('Evaluation EMA model...') if rank == 0 else None
        mDICE_ema, dice_class_ema = evaluate(model_ema, valloader, eval_mode, cfg, rank, multiplier=cfg['patch_size'])
        
        if rank == 0:
            for cls_idx, dice in enumerate(dice_class):
                print('RESULTS | Class [{:} {:}] DICE: {:.3f}, '
                            'EMA: {:.3f}'.format(cls_idx+1, list(LABEL_PROJ_DICT.keys())[cls_idx], dice, dice_class_ema[cls_idx]))
            print(' Evaluation {} Total Results >>>> mDICE: {:.3f}, EMA mDICE: {:.3f}\n'.format(eval_mode, mDICE, mDICE_ema))
            
            writer.add_scalar('eval/mDICE', mDICE, epoch)
            writer.add_scalar('eval/mDICE_ema', mDICE_ema, epoch)
            for i, dice in enumerate(dice_class):
                writer.add_scalar('eval/%s_DICE' % (list(LABEL_PROJ_DICT.keys())[i]), dice, epoch)
                writer.add_scalar('eval/%s_DICE_ema' % (list(LABEL_PROJ_DICT.keys())[i]), dice_class_ema[i], epoch)

        is_best = mDICE >= previous_best
        
        previous_best = max(mDICE, previous_best)
        previous_best_ema = max(mDICE_ema, previous_best_ema)
        if mDICE == previous_best:
            best_epoch = epoch
        if mDICE_ema == previous_best_ema:
            best_epoch_ema = epoch
        
        if rank == 0:
            print(f'Final Results: Best epoch {best_epoch} with mDICE {previous_best:.3f},Best EMA epoch {best_epoch_ema} with mDICE {previous_best_ema:.3f}\n')
            checkpoint = {
                'model': model.state_dict(),
                'model_ema': model_ema.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'previous_best': previous_best,
                'previous_best_ema': previous_best_ema,
                'best_epoch': best_epoch,
                'best_epoch_ema': best_epoch_ema
            }
            torch.save(checkpoint, os.path.join(args.save_path, 'latest.pth'))
            if is_best:
                torch.save(checkpoint, os.path.join(args.save_path, 'best.pth'))


if __name__ == '__main__':
    main()
