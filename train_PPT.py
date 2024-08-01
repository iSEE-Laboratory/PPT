import argparse
from trainer import trainer_PPT as trainer_ppt
import logging
import random, os, torch
import numpy as np

def seed_torch(seed=1666):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def parse_config():
    parser = argparse.ArgumentParser(description='MemoNet with SDD dataset')

    parser.add_argument("--cuda", default=True)
    # verify the CUDA_VISIBLE_DEVICES
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument("--seed", type=int, default=1666)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--learning_rate_des", type=float, default=0.000001)
    parser.add_argument("--max_epochs", type=int, default=600)

    parser.add_argument("--past_len", type=int, default=8, help="length of past (in timesteps)")
    parser.add_argument("--future_len", type=int, default=12, help="length of future (in timesteps)")
    parser.add_argument("--dim_embedding_key", type=int, default=128)

    # Configuration for SDD dataset.
    parser.add_argument("--dataset_name", type=str, default='sdd')
    parser.add_argument("--data_scene", type=str, default='eth')
    parser.add_argument("--data_scale", type=float, default=1)
    parser.add_argument("--data_scale_old", type=float, default=1.86)
    parser.add_argument("--train_b_size", type=int, default=512)
    parser.add_argument("--test_b_size", type=int, default=4096)
    parser.add_argument("--time_thresh", type=int, default=0)
    parser.add_argument("--dist_thresh", type=int, default=100)

    # Transformer Decoder Param
    parser.add_argument("--n_layer", type=int, default=2)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--n_embd", type=int, default=128)
    parser.add_argument("--T", type=int, default=8)
    # parser.add_argument("--layer_num", type=int, default=4)
    parser.add_argument('--num_1_list', nargs='+', type=int, default=[2, 2, 2])
    # parser.add_argument('--num_1_list', type=list, nargs='+', default=[2, 2, 2])
    parser.add_argument("--vocab_size", type=int, default=2)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--embd_pdrop", type=float, default=0.1)
    parser.add_argument("--resid_pdrop", type=float, default=0.1)
    parser.add_argument("--attn_pdrop", type=float, default=0.1)

    # Loss Function
    parser.add_argument("--lambda_j", type=float, default=100)
    parser.add_argument("--lambda_recon", type=float, default=1)
    parser.add_argument("--d_scale", type=float, default=1)
    parser.add_argument("--lambda_vec", type=float, default=0.0005)
    parser.add_argument("--lambda_vel", type=float, default=0.01)
    parser.add_argument("--lambda_des", type=float, default=4)
    parser.add_argument("--lambda_desloss", type=float, default=1)
    parser.add_argument("--traj_lambda_soft", type=float, default=0.3)
    parser.add_argument("--trajp_lambda_soft", type=float, default=0.1)
    parser.add_argument("--des_lambda_soft", type=float, default=0.1)

    parser.add_argument("--mode", type=str, default='Short_term', choices=['Short_term', 'Des_warm', 'Long_term', 'ALL'], help='Stage of training.')
    parser.add_argument("--model_Pretrain", default='./training/training_ae/model_encdec')
    parser.add_argument("--model_ST", default=None)
    parser.add_argument("--model_LT", default=None)
    parser.add_argument("--info", type=str, default='', help='Name of training. '
                                                             'It will be used in test folder')
    return parser.parse_args()


def main(config):
    seed_torch(config.seed)
    t = trainer_ppt.Trainer(config)
    t.logger.info('[M] start training modules for SDD dataset.')
    t.fit()


if __name__ == "__main__":
    config = parse_config()
    print(config)
    main(config)
