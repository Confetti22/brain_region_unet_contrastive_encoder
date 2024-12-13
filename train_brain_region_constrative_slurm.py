import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch

import argparse
from lib.utils.file import bool_flag
from lib.utils.distributed import init_dist_node, init_dist_gpu, get_shared_folder


import submitit, random, sys
from pathlib import Path


def parse_args():

    parser = argparse.ArgumentParser(description='Template')

    # === PATHS === #
    parser.add_argument('-data', type=str, default="data",
                                            help='path to dataset directory')
    parser.add_argument('-out', type=str, default="out",
                                            help='path to out directory')

    # === GENERAL === #
    parser.add_argument('-model', type=str, default="recon_smallnet_sme_loss_woreg_256dim",
                                            help='Model name')
    parser.add_argument('-reset', action='store_true',
                                            help='Reset saved model logs and weights')
    parser.add_argument('-tb', action='store_true',
                                            help='Start TensorBoard')
    parser.add_argument('-gpus', type=str, default="0",
                                            help='GPUs list, only works if not on slurm')
    parser.add_argument('-cfg', type =str,help='Configuration file',
                        default='config/brain_region_unet.yaml')

    # === Dataset === #
    parser.add_argument('-dataset', type=str, default = 'random',
                                            help='Dataset to choose')
    # parser.add_argument('-dataset_name',type=str,defalut='cifar10', 
    #                                         help='the name of the dataset')
    parser.add_argument('-batch_per_gpu', type=int, default = 2,
                                            help='batch size per gpu')
    parser.add_argument('-shuffle', type=bool_flag, default = True,
                                            help='Shuffle dataset')
    parser.add_argument('-workers', type=int, default = 2,
                                            help='number of workers')

    # === Architecture === #
    parser.add_argument('-arch', type=str, default = 'mlp',
                                            help='Architecture to choose')

    # === Trainer === #
    parser.add_argument('-trainer', type=str, default = 'trainer',
                                            help='Trainer to choose')
    parser.add_argument('-epochs', type=int, default = 5000,
                                            help='number of epochs')
    parser.add_argument('-save_every', type=int, default = 10,
                                            help='Save frequency')
    parser.add_argument('-fp16', type=torch.dtype,default=torch.float32, help='bfloat16 will be more numerical stable')


    # === Optimization === #
    parser.add_argument('-optimizer', type=str, default = 'adam',
                                            help='Optimizer function to choose')
    parser.add_argument('-lr_start', type=float, default = 5e-4,
                                            help='Initial Learning Rate')
    parser.add_argument('-lr_end', type=float, default = 1e-6,
                                            help='Final Learning Rate')
    parser.add_argument('-lr_warmup', type=int, default = 10,
                                            help='warmup epochs for learning rate')

    # === SLURM === #
    parser.add_argument('-slurm', action='store_true',
                                            help='Submit with slurm')
    parser.add_argument('-slurm_ngpus', type=int, default = 8,
                                            help='num of gpus per node')
    parser.add_argument('-slurm_nnodes', type=int, default = 2,
                                            help='number of nodes')
    parser.add_argument('-slurm_nodelist', default = None,
                                            help='slurm nodeslist. i.e. "GPU17,GPU18"')
    parser.add_argument('-slurm_partition', type=str, default = "general",
                                            help='slurm partition')
    parser.add_argument('-slurm_timeout', type=int, default = 2800,
                                            help='slurm timeout minimum, reduce if running on the "Quick" partition')


    args = parser.parse_args()

    # cmdline parameters will overwrite the CFG parameters

    # === Read CFG File === #
    if args.cfg:
        with open(args.cfg, 'r') as f:
            import yaml
            yml = yaml.safe_load(f)

        # update values from cfg file only if not passed in cmdline
        cmd = [c[1:] for c in sys.argv if c[0]=='-']
        for k,v in yml.items():
            if k not in cmd:
                args.__dict__[k] = v

    return args


class SLURM_Trainer(object):
    def __init__(self, args,cfg):
        self.args = args
        self.cfg=cfg

    def __call__(self):

        init_dist_node(self.args)
        train(None, self.args, self.cfg)


from config import load_cfg
def main():

    args = parse_args()
    args.port = random.randint(49152,65535)

    cfg=load_cfg(args)
    
    if args.slurm:

        # Almost copy-paste from https://github.com/facebookresearch/deit/blob/main/run_with_submitit.py
        args.output_dir = get_shared_folder(args) / "%j"
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        executor = submitit.AutoExecutor(folder=args.output_dir, slurm_max_num_timeout=30)

        executor.update_parameters(
            mem_gb=128*args.slurm_ngpus,
            gpus_per_node=args.slurm_ngpus,
            tasks_per_node=args.slurm_ngpus,
            cpus_per_task=2,
            nodes=args.slurm_nnodes,
            timeout_min=2800,
            slurm_partition=args.slurm_partition
        )

        if args.slurm_nodelist:
            executor.update_parameters(slurm_additional_parameters = {"nodelist": f'{args.slurm_nodelist}' })

        executor.update_parameters(name=args.model)
        trainer = SLURM_Trainer(args,cfg)
        job = executor.submit(trainer)
        print(f"Submitted job_id: {job.job_id}")


    else:
        init_dist_node(args)
        mp.spawn(train, args=(args,cfg), nprocs=args.ngpus_per_node)
	

def train(gpu, args, cfg):


    # === SET ENV === #
    init_dist_gpu(gpu, args)
    
    # === DATA === #
    # get_dataset = getattr(__import__("lib.datasets.{}".format(args.dataset), fromlist=["get_dataset"]), "get_dataset")
    from lib.datasets.visor_3d_dataset import get_dataset
    dataset = get_dataset(cfg)

    sampler = DistributedSampler(dataset, shuffle=cfg.DATASET.shuffle, num_replicas = args.world_size, rank = args.rank, seed = 31)
    print(f"batch_size {cfg.DATASET.batch_per_gpu} ")
    loader = DataLoader(dataset=dataset, 
                            sampler = sampler,
                            batch_size=cfg.DATASET.batch_per_gpu, 
                            num_workers= cfg.DATASET.num_workers,
                            pin_memory = True,
                            drop_last = True
                            )
    print(f"Data loaded")

    # === MODEL === #
    from lib.arch.autoencoder import build_autoencoder_model
    from torchsummary import summary

    model = build_autoencoder_model(cfg)
    model=model.cuda(args.gpu)
    model.train()

    
    #print out model info
    print(model)
    summary(model,(1,128,128,128))

    # model = nn.SyncBatchNorm.convert_sync_batchnorm(model) #group_norm did not require to sync, group_norm is preferred when batch_size is small
    model = nn.parallel.DistributedDataParallel(model, device_ids= [cfg.SYSTEM.GPU_IDS])



    # === LOSS === #
    from lib.core.contrastive_loss import get_loss
    contrastive_loss = get_loss(cfg).cuda(args.gpu)
    
    #reconsturct the preprocess image in range [0-1] with bce loss
    recon_loss = nn.MSELoss()

    # === OPTIMIZER === #
    from lib.core.optimizer import get_optimizer
    # optimizer = get_optimizer(model, args)
    optimizer = torch.optim.Adam(model.parameters(),lr = args.lr_start)

    # === TRAINING === #
    Trainer = getattr(__import__("lib.trainers.{}".format(args.trainer), fromlist=["Trainer"]), "Trainer")
    Trainer(args, loader, model, recon_loss, optimizer).fit()


if __name__ == "__main__":
    main()
