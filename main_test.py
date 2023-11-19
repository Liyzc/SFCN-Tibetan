import argparse
import os
import sys
from models.VGG import *
import data_loaders
from data_loaders import cifar10, tmnist
from functions import *
from utils import val
from models import *
import attack
import copy
import torch
import json
import wandb

parser = argparse.ArgumentParser()
# just use default setting
parser.add_argument('-j','--workers',default=1, type=int,metavar='N',help='number of data loading workers')
parser.add_argument('-b','--batch_size',default=64, type=int,metavar='N',help='mini-batch size')
parser.add_argument('-sd', '--seed',default=42,type=int,help='seed for initializing training.')
parser.add_argument('-suffix','--suffix',default='', type=str,help='suffix')
parser.add_argument('--wandb',default='b87c5458732c2068da47f3eca9f46f4ab8c00d68', type=str,help='wandb key')
parser.add_argument('--name',default='tbc', type=str,help='wandb run name')

# model configuration
parser.add_argument('-data', '--dataset', default='tmnist',type=str,help='dataset')
parser.add_argument('-arch','--model', default='AFCN', type=str,help='model')
parser.add_argument('-T','--time', default=0, type=int, metavar='N',help='snn simulation time')
parser.add_argument('-id', '--identifier', type=str, help='model statedict identifier')
parser.add_argument('-config', '--config', default='', type=str,help='test configuration file')

# training configuration
parser.add_argument('-dev','--device',default='0,1',type=str,help='device')

# adv atk configuration
parser.add_argument('-atk','--attack',default=' ',type=str,help='attack')
parser.add_argument('-eps','--eps',default=2,type=float,metavar='N',help='attack eps')
parser.add_argument('-atk_m','--attack_mode',default=' ', type=str,help='attack mode')

# only pgd
parser.add_argument('-alpha','--alpha',default=2.55/1,type=float,metavar='N',help='pgd attack alpha')
parser.add_argument('-steps','--steps',default=7,type=int,metavar='N',help='pgd attack steps')
parser.add_argument('-bb','--bbmodel',default='',type=str,help='black box model') # vgg11_clean_l2[0.000500]bb
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

name = '-'.join(['test', args.model, args.name, args.attack, args.attack_mode])
os.environ["WANDB_API_KEY"] = args.wandb 
os.environ["WANDB_MODE"] = "offline"
run = wandb.init(
    project="tibetan-SNN",
    entity="lynne",
    name=name,
    tags=['ntest', args.model, args.name, args.attack, args.attack_mode],
)

def main():
    global args
    if args.dataset.lower() == 'cifar10':
        use_cifar10 = True
        num_labels = 10
    elif args.dataset.lower() == 'cifar100':
        use_cifar10 = False
        num_labels = 100
    elif args.dataset.lower() == 'svhn':
        num_labels = 10
    elif args.dataset.lower() == 'tmnist':
        num_labels = 10
        # print("11111")

    # log_dir = '%s-results'% (args.dataset)
    log_dir = '%s-checkpoints'% (args.dataset)
    model_dir = '%s-checkpoints'% (args.dataset)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    logger = get_logger(os.path.join(log_dir, '%s.log'%(args.identifier+args.suffix)))
    logger.info('start testing!')

    seed_all(args.seed)
    if 'cifar' in args.dataset.lower():
        _, val_dataset, znorm = data_loaders.build_cifar(use_cifar10=use_cifar10)
    elif args.dataset.lower() == 'svhn':
        _, val_dataset, znorm = data_loaders.build_svhn()
    elif args.dataset.lower() == 'tmnist':
        _, val_dataset, znorm = tmnist()
    test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    if 'vgg_wobn' in args.model.lower():
        model = VGG_woBN(args.model.lower(), args.time, num_labels, znorm)
    elif 'vgg' in args.model.lower():
        model = VGG(args.model.lower(), args.time, num_labels, znorm)
    elif 'wideresnet' in args.model.lower():
        model = WideResNet(args.model.lower(), args.time, num_labels, znorm)
    elif 'sfcn' in args.model.lower():
        model = SFCN(args.model.lower(), args.time, num_labels, znorm)
    elif 'afcn' in args.model.lower():
        model = AFCN(args.model.lower(), args.time, num_labels, znorm)
    else:
        raise AssertionError("model not supported")

    model.set_simulation_time(args.time)
    model.to(device)

    # have bb model
    if len(args.bbmodel) > 0:
        bbmodel = copy.deepcopy(model)
        bbstate_dict = torch.load(os.path.join(model_dir, args.bbmodel+'.pth'), map_location=torch.device('cpu'))
        bbmodel.load_state_dict(bbstate_dict, strict=False)
    else:
        bbmodel = None

    if len(args.config) > 0:
        with open(args.config+'.json', 'r') as f:
            config = json.load(f)
    else:
        config = [{}]
    for atk_config in config:
#         logger.info(json.dumps(atk_config))
        state_dict = torch.load(os.path.join(model_dir, args.identifier + '.pth'), map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        model.to(device)

        for arg in atk_config.keys():
            setattr(args, arg, atk_config[arg])
        if 'bb' in atk_config.keys() and atk_config['bb']:
            atkmodel = bbmodel
        else:
            atkmodel = model

        if args.attack_mode == 'bptt':
            ff = BPTT_attack
        elif args.attack_mode == 'bptr':
            ff = BPTR_attack
        else:
            ff = Act_attack

        if args.eps == -1:
            eps_list = [0,0.02,0.04,0.06,0.08,0.1]
        else:
            eps_list = [args.eps]

        t_pgd = 8

        for eps in eps_list:
            if args.attack.lower() == 'fgsm':
                atk = attack.FGSM(atkmodel, forward_function=ff, eps=eps, T=args.time)
            elif args.attack.lower() == 'pgd':
                atk = attack.PGD(atkmodel, forward_function=ff, eps=eps, alpha=args.alpha / 255, steps=args.steps, T=t_pgd, device=device)
            elif args.attack.lower() == 'gn':
                atk = attack.GN(atkmodel, forward_function=ff, eps=eps, T=args.time)
            else:
                atk = None

            acc = val(model, test_loader, device, args.time, atk)
            logger.info(json.dumps(atk_config)+' Test acc={:.3f}'.format(acc))
            wandb.log({'eps': eps, 'Eval Acc': acc})
            atk = None
            # print("here")


if __name__ == "__main__":
    main()
