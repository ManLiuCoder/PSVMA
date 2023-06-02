import os
from os.path import join
import argparse
import torch
import numpy as np
import random
from data import build_dataloader
from models.modeling import build_gzsl_pipeline
from models.config import cfg
from models.utils.comm import *
from models.utils import ReDirectSTD
from models.engine.inferencer import eval_zs_gzsl



def test_model(cfg, local_rank, distributed):
    model = build_gzsl_pipeline(cfg) 
    model_dict = model.state_dict()
    saved_dict = torch.load('checkpoints/best_model_cub.pth')
    saved_dict = {k: v for k, v in saved_dict['model'].items() if k in model_dict}
    model_dict.update(saved_dict)
    model.load_state_dict(model_dict)

    device = torch.device(cfg.MODEL.DEVICE)
    model = model.to(device)


    tr_dataloader, tu_loader, ts_loader, res = build_dataloader(cfg, is_distributed=distributed)


    test_gamma = cfg.TEST.GAMMA

    acc_seen, acc_novel, H, acc_zs = eval_zs_gzsl(
            tu_loader,
            ts_loader,
            res,
            model,
            test_gamma,
            device)
    print('zsl: %.4f, gzsl: seen=%.4f, unseen=%.4f, h=%.4f' % (acc_zs, acc_seen, acc_novel, H))

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
    model = test_model(cfg, args.local_rank, args.distributed)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    main()