from __future__ import print_function
import numpy as np
import argparse, os, time, random
from tqdm import tqdm
import logging
import torch, torchvision
import torch.backends.cudnn as cudnn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision.datasets import *
from replace import clip
from models import prompters
from models.prompters import TokenPrompter,NullPrompter
from models.model import *
from attacks import *
import copy
from utils import accuracy, AverageMeter, ProgressMeter, save_checkpoint
from utils import cosine_lr, convert_models_to_fp32, refine_classname
from utils import load_train_dataset, load_val_datasets, get_text_prompts_train, \
    get_text_prompts_val

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

def parse_option():
    parser = argparse.ArgumentParser('Adapting CLIP for zero-shot adv robustness')
    parser.add_argument('--print_freq', type=int, default=50, help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50, help='save frequency')
    parser.add_argument('--validate_freq', type=int, default=9, help='validate frequency')
    parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=1, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=10, help='number of training epochs')

    # optimization
    parser.add_argument('--optim', type=str, default='EText', help='optimizer to use')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
    parser.add_argument("--weight_decay", type=float, default=0, help="weight decay")
    parser.add_argument("--warmup", type=int, default=1000, help="number of steps to warmup for")
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    
    # adversarial attack
    parser.add_argument('--train_eps', type=float, default=1, help='momentum')
    parser.add_argument('--train_numsteps', type=int, default=2)
    parser.add_argument('--train_stepsize', type=int, default=1)
    parser.add_argument('--test_eps', type=float, default=1, help='momentum')
    parser.add_argument('--test_numsteps', type=int, default=100)
    parser.add_argument('--test_stepsize', type=int, default=1)

    # model
    parser.add_argument('--patience', type=int, default=1000)
    parser.add_argument('--model', type=str, default='clip')
    parser.add_argument('--arch', type=str, default='vit_b32')
    parser.add_argument('--method', type=str, default='null_patch',
                        choices=['padding', 'random_patch', 'fixed_patch', 'null_patch'],
                        help='choose visual prompting method')

    # dataset
    parser.add_argument('--root', type=str, default='./data', help='dataset')
    parser.add_argument('--dataset', type=str, default='tinyImageNet',
                        choices=['cifar100', 'ImageNet', 'cifar10', 'tinyImageNet'],
                        help='dataset for training')
    parser.add_argument('--image_size', type=int, default=224, help='image size')


    # other
    parser.add_argument('--seed', type=int, default=1, help='seed for initializing training')
    parser.add_argument('--model_dir', type=str, default='./save/models', help='path to save models')
    parser.add_argument('--filename', type=str, default=None, help='filename to save')
    parser.add_argument('--trial', type=int, default=1, help='number of trials')
    parser.add_argument('--resume', type=str, default=None, help='path to resume from checkpoint')


    parser.add_argument('--gpu', type=int, default=0, help='gpu to use')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--VPbaseline', action='store_true')
    parser.add_argument('--attack', choices=['pgd', 'CW', 'autoattack'], default='pgd')
    parser.add_argument('--noimginprop', action='store_true')
    
    #FT
    parser.add_argument('--last_num_ft', type=int, default=0)
    parser.add_argument('--adaptation_method', type=str, default='FT', choices=['VPT','FT'],
                        help='choose visual adaptation method')
    parser.add_argument('--Distance_metric', type=str, default='l2', choices=['cos', 'l2', 'KL', 'l1'],
                        help='Select the distance measure in the loss function')
    parser.add_argument('--Alpha', type=float, default=0.03,help='loss 1')
    parser.add_argument('--Beta', type=float, default=0.07, help='loss 2')
    parser.add_argument('--gamma', type=int, default=900, help='random texts number')
    parser.add_argument('--testdata', type=str, nargs='+')
    args = parser.parse_args()

    args.filename = '{}_{}_{}_{}_{}_lr-{}_decay-{}_bsz-{}_warmup-{}_trial-{}_Alpha-{}_Beta-{}_gamma-{}_distance-{}'. \
        format(args.adaptation_method,args.dataset, args.model, args.arch,
               args.optim, args.learning_rate, args.weight_decay, args.batch_size, args.warmup, args.trial,
            args.Alpha, args.Beta, args.gamma, args.Distance_metric)
    return args
    

def main():
    global best_acc1, device, logger
    args = parse_option()
    device = torch.device("cuda:{}".format(args.gpu))
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    log_dir = './save/loggers/'
    os.makedirs(log_dir, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(log_dir,f'{args.filename}.log'))
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(filename)s] => %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
   
    args.train_eps = args.train_eps / 255.
    args.test_eps = args.test_eps / 255.
    args.train_stepsize = args.train_stepsize / 255.
    args.test_stepsize = args.test_stepsize / 255.

    args_dict = vars(args)
    for key, value in args_dict.items():
        print(f'{key}: {value}')
        logger.info(f'{key}: {value}')


    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    """create model"""
    if args.adaptation_method == 'VPT':
        add_prompt_len = args.add_prompt_size
    else:
        add_prompt_len = 0
    print(" create model")
    model, preprocess = clip.load('ViT-B/32', device, jit=False, prompt_len=add_prompt_len)

    convert_models_to_fp32(model)
    model = model.to(device)
    frozen_model = copy.deepcopy(model).to(device)
    
    model.eval()
    frozen_model.eval() 
    
    """define criterion and optimizer"""
    if args.adaptation_method == 'VPT':
        prompter = prompters.__dict__[args.method](args).to(device)
        add_prompter = TokenPrompter(args.add_prompt_size).to(device)
        optimizer = torch.optim.SGD(list(prompter.parameters()) + list(add_prompter.parameters()),
                                lr=args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    else:
        prompter = NullPrompter().to(device)
        add_prompter = TokenPrompter(0).to(device)
        if args.last_num_ft == 0:
            optimizer = torch.optim.SGD(model.visual.parameters(),
                                        lr=args.learning_rate,
                                        momentum=args.momentum,
                                        weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.SGD(list(model.visual.parameters())[-args.last_num_ft:],
                                        lr=args.learning_rate,
                                        momentum=args.momentum,
                                        weight_decay=args.weight_decay)
    
    """Load the pre-trained model"""
    args.start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']

            if 'vision_encoder_state_dict' in checkpoint.keys():
                model.visual.load_state_dict(checkpoint['vision_encoder_state_dict'], strict=False)
            else:
                prompter.load_state_dict(checkpoint['state_dict'])
                add_prompter.load_state_dict(checkpoint['add_prompter'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
            logger.info("loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    template = 'This is a photo of a {}'
    print(f'template: {template}')


    """load training dataset"""
    train_dataset = load_train_dataset(args)

    """load val dataset(s)"""
    if args.testdata is None:
        val_dataset_name = ['tinyImageNet','cifar10', 'cifar100','STL10','Food101','oxfordpet','flowers102','dtd','EuroSAT',\
                            'fgvc_aircraft','Caltech101','Caltech256','StanfordCars','PCAM','ImageNet','SUN397']
    else:
        val_dataset_name = args.testdata
    val_dataset_list = load_val_datasets(args, val_dataset_name)


    """create dataloaders"""
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                               shuffle=True, worker_init_fn=np.random.seed(args.seed), generator=torch.Generator().manual_seed(args.seed))

    val_loader_list = [DataLoader(each, batch_size=args.batch_size*2, 
                                   shuffle=False) for each in val_dataset_list]

    """get text prompts for training/val"""
    texts_train = get_text_prompts_train(args, train_dataset, template=template)
    texts_list = get_text_prompts_val(val_dataset_list, val_dataset_name, template=template)

    scaler = GradScaler()
    total_steps = len(train_loader) * args.epochs
    scheduler = cosine_lr(optimizer, args.learning_rate, args.warmup, total_steps)

    cudnn.benchmark = True

    args.model_folder = os.path.join(args.model_dir, args.filename)
    if not os.path.isdir(args.model_folder):
        os.makedirs(args.model_folder)

    epochs_since_improvement = 0
    best_acc1 = 0.
    """training"""
    for epoch in range(args.start_epoch, args.epochs):
        train(train_loader, texts_train, model,frozen_model, prompter, add_prompter, optimizer, scheduler, scaler, epoch,  args)
    validate(val_loader_list, val_dataset_name, texts_list, model,frozen_model,optimizer, device,
                                prompter, add_prompter, args)



"""train function"""
def train(train_loader, texts, model,frozen_model, prompter, add_prompter,
          optimizer, scheduler, scaler, epoch,  args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1],
        prefix="Epoch: [{}]".format(args.start_epoch + epoch))

    """switch to train mode"""
    prompter.train()
    add_prompter.train()
    model.visual.train()
    num_batches_per_epoch = len(train_loader)

    alpha = args.train_stepsize
    attack_iters = args.train_numsteps
    text_tokens = clip.tokenize(texts).to(device)
    end = time.time()
    for i, (images, target) in enumerate(tqdm(train_loader)):

        # measure data loading time
        data_time.update(time.time() - end)
        BATCH_SIZE = images.size(0)

        # adjust learning rate
        step = num_batches_per_epoch * epoch + i
        scheduler(step)

        optimizer.zero_grad()

        images = images.to(device)
        target = target.to(device)


        # with automatic mixed precision
        with autocast():
            with torch.no_grad():
                logit_scale=model.logit_scale.exp()
                text_features = model.encode_text(text_tokens)
                text_features = text_features / text_features.norm(dim=1,keepdim=True)
                text_embed = logit_scale * text_features
                
            """Build adversarial example"""
            if not args.VPbaseline:
                delta = attack_pgd(prompter, model,add_prompter,images,
                                target, text_tokens, alpha, attack_iters, 'l_inf',
                                device=device, args=args, epsilon=args.train_eps)
                tmp = clip_img_preprocessing(images + delta,device)
            else:
                tmp = clip_img_preprocessing(images,device)

            prompted_images = prompter(tmp)
            clean_images = prompter(clip_img_preprocessing(images,device))
            prompt_token = add_prompter()

            adv_features = model.encode_image(prompted_images, prompt_token)[:,0,:]  # finetuning model; attack
            adv_features = adv_features / adv_features.norm(dim=1, keepdim=True)
            output = adv_features @ text_embed.t()

            ori_features = frozen_model.encode_image(clean_images, prompt_token)[:,0,:] # frozen model; clean
            ori_features = ori_features / ori_features.norm(dim=1, keepdim=True)

            
            cle_features = model.encode_image(clean_images, prompt_token)[:,0,:] # finetuning model; clean
            cle_features = cle_features / cle_features.norm(dim=1, keepdim=True)

            
            CrossEntropyLoss = torch.nn.CrossEntropyLoss().to(device)
            loss_TeCoA = CrossEntropyLoss(output, target)


            random_output_fro, random_output, random_text_features = random_dis(i, cle_features, ori_features, model, text_tokens, device, args)
            output1 = adv_features @ random_text_features.t()
            output2 = ori_features @ random_text_features.t()

            loss_MSE1 = args.Alpha * torch.mean(torch.norm(output1 - output2,dim=1, p=2))
            loss_MSE2 = args.Beta * torch.mean(torch.norm(random_output_fro - random_output,dim=1, p=2))

            loss = loss_TeCoA + loss_MSE1 + loss_MSE2
            scaler.scale(loss).backward()
            scaler.step(optimizer)
        scaler.update()

        model.logit_scale.data = torch.clamp(model.logit_scale.data, 0, 4.6052)   
        # measure accuracy
        acc1 = accuracy(output, target, topk=(1,))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0].item(), images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            entries = progress.display(i)
            logger.info(entries)
            logger.info("TeCoA Loss: %f, MSE1 Loss: %f, MSE2 Loss: %f", loss_TeCoA, loss_MSE1, loss_MSE2)
            print("TeCoA Loss: {:.3f}, MSE1 Loss: {:.3f}, MSE2 Loss: {:.3f}".format(loss_TeCoA.item(), loss_MSE1.item(), loss_MSE2.item()))
            if args.debug:
                break
    save_checkpoint({
        'epoch':  args.start_epoch + epoch + 1,
        'state_dict': prompter.state_dict(),
        'add_prompter': add_prompter.state_dict(),
        'best_acc1': best_acc1,
        'optimizer': optimizer.state_dict(),
        'vision_encoder_state_dict':model.visual.state_dict(),
        }, args)
    return losses.avg, top1.avg


def validate(val_loader_list, val_dataset_name, texts_list, model,frozen_model,optimizer, device,
                prompter, add_prompter, args):
    dataset_num = len(val_loader_list)
    acc_all = []
    test_stepsize = args.test_stepsize
    for cnt in range(dataset_num):

        val_loader = val_loader_list[cnt]
        texts = texts_list[cnt]
        dataset_name = val_dataset_name[cnt]

        binary = ['PCAM', 'hateful_memes']
        attacks_to_run=['apgd-ce', 'apgd-dlr']
        if dataset_name in binary:
            attacks_to_run=['apgd-ce']
            
        batch_time = AverageMeter('Time', ':6.3f')
        top1_org = AverageMeter('Original Acc@1', ':6.2f')
        top1_adv_org = AverageMeter('Adv Original Acc@1', ':6.2f')

        progress = ProgressMeter(
            len(val_loader),
            [batch_time, top1_org, top1_adv_org],
            prefix=dataset_name + '_Validate: ')

        # switch to evaluation mode
        prompter.eval()
        add_prompter.eval()
        model.eval()
        model.zero_grad()
        frozen_model.eval()
        
        text_tokens = clip.tokenize(texts,truncate=True).to(device)
        end = time.time()
        for i, (images, target) in enumerate(tqdm(val_loader)):
            images = images.to(device)
            target = target.to(device)
            
            with autocast():
                # compute output
                with torch.no_grad():
                    """clean images"""
                    prompt_token = None
                    output_org, _, text_features = multiGPU_CLIP(model, clip_img_preprocessing(images,device),text_tokens,target, device, None)
                    acc1 = accuracy(output_org, target, topk=(1,))
                    top1_org.update(acc1[0].item(), images.size(0))


                """adv images"""
                if args.attack == 'CW':
                    delta_noprompt = attack_CW(None, model, None,  images, target, text_tokens,
                                        test_stepsize, args.test_numsteps, 'l_inf',device, args, epsilon=args.test_eps)
                    attacked_images = images + delta_noprompt
                elif args.attack == 'pgd':
                    delta_noprompt = attack_pgd(None, model, None, images, target, text_tokens,
                                        test_stepsize, args.test_numsteps,'l_inf',device, args, epsilon=args.test_eps)
                    attacked_images = images + delta_noprompt
                else:
                    attacked_images  = attack_auto(model, images, target, text_tokens, None, None, device,
                                            attacks_to_run=attacks_to_run, epsilon=args.test_eps)
                    
                # torch.cuda.empty_cache()
                with torch.no_grad():
                    output_org_adv= multiGPU_CLIP(model, clip_img_preprocessing(attacked_images,device),
                                                        text_tokens, target, device, None)[0]
                    acc1 = accuracy(output_org_adv, target, topk=(1,))
                    top1_adv_org.update(acc1[0].item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                entries = progress.display(i)
                logger.info(entries)
                if args.debug:
                    break


        print(dataset_name + ' * Adv Original Acc@1 {top1_adv_org.avg:.3f} '
                             '*  Original Acc@1 {top1_org.avg:.3f}'
              .format(top1_adv_org=top1_adv_org,
                      top1_org=top1_org))
        logger.info(dataset_name + ' * Adv Original Acc@1 {top1_adv_org.avg:.3f} '
                             '* Original Acc@1 {top1_org.avg:.3f}'
              .format(top1_adv_org=top1_adv_org, top1_org=top1_org))
        
        acc_all.append(top1_adv_org.avg)

    return np.mean(acc_all)


def random_dis(batch, cle_features, ori_features, model, text_tokens, device, args):
    img_embed_fro = cle_features # frozen model; clean
    img_embed = ori_features # finetuning model; clean
    """random texts"""
    text_num = args.gamma
    torch.manual_seed(args.seed + batch)
    random_texts = torch.randint(low=0, high=49406, size=(text_num, 77)).to(device)
    random_texts[:, 0] = 49406
    random_texts[:,-1] = 49407    
    text_features = torch.tensor([]).to(device)
    with torch.no_grad():
        for i in range(text_num // 100):
            text_feature = model.encode_text(random_texts[100*i:100*(i+1),:])
            text_features = torch.cat((text_features, text_feature), dim=0)
        text_features = torch.linalg.qr(text_features)[0]
        # print(text_features.size())
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        scale_text_embed = model.logit_scale.exp() * text_features
    output_fro = img_embed_fro @ scale_text_embed.t()
    output = img_embed @ scale_text_embed.t()
    # output_fro = img_embed_fro
    # output = img_embed
    return output_fro, output, scale_text_embed

if __name__ == '__main__':
    main()
