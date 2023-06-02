import os
from os.path import join
import argparse
import torch
import numpy as np
import random
from data import build_dataloader
from models.modeling import build_gzsl_pipeline
from models.solver import make_optimizer, make_lr_scheduler
from models.engine.trainer import do_train
from models.config import cfg
from models.utils.comm import *
from models.utils import ReDirectSTD

try:
    from apex import amp
except ImportError:
    raise ImportError('Use APEX for multi-precision via apex.amp')

def train_model(cfg, local_rank, distributed):
    seed = 12345
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = build_gzsl_pipeline(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model = model.to(device)

    optimizer = make_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer)

    use_mixed_precision = cfg.DTYPE == "float16"
    amp_opt_level = 'O1' if use_mixed_precision else 'O0'
    model, optimizer = amp.initialize(model, optimizer, opt_level=amp_opt_level)
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            broadcast_buffers=False,
        )

    tr_dataloader, tu_loader, ts_loader, res = build_dataloader(cfg, is_distributed=distributed)

    output_dir = cfg.OUTPUT_DIR
    model_file_name = cfg.MODEL_FILE_NAME
    model_file_path = join(output_dir, model_file_name)

    test_gamma = cfg.TEST.GAMMA
    max_epoch = cfg.SOLVER.MAX_EPOCH

    lamd = {
        1: cfg.MODEL.LOSS.LAMBDA1,
        2: cfg.MODEL.LOSS.LAMBDA2,
        3: cfg.MODEL.LOSS.LAMBDA3,
    }

    do_train(
        model,
        tr_dataloader,
        tu_loader,
        ts_loader,
        res,
        optimizer,
        scheduler,
        lamd,
        test_gamma,
        device,
        max_epoch,
        model_file_path,
    )

    return model


def main():
    parser = argparse.ArgumentParser(description="PyTorch Zero-Shot Learning Training")
    parser.add_argument(
        "--config-file",
        default="config/sun.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.freeze()


    output_dir = cfg.OUTPUT_DIR
    log_file_name = cfg.LOG_FILE_NAME

    log_file_path = join(output_dir, log_file_name)

    if is_main_process():
        ReDirectSTD(log_file_path, 'stdout', True)

    print("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        print(config_str)
    print("Running with config:\n{}".format(cfg))
        
    torch.backends.cudnn.benchmark = True
    model = train_model(cfg, args.local_rank, args.distributed)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    main()