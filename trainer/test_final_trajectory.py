import os
import datetime

import numpy as np
import torch
import torch.nn as nn
from models.model_test_trajectory import Final_Model
from torch.utils.data import DataLoader

from dataset_loader import *
from vis_ethucy import *

torch.set_num_threads(5)

class Trainer:
    def __init__(self, config):

        # test folder creating
        self.name_test = str(datetime.datetime.now())[:10]
        self.folder_test = 'testing/' + self.name_test + '_' + config.info
        if not os.path.exists(self.folder_test):
            os.makedirs(self.folder_test)
        self.folder_test = self.folder_test + '/'

        # Initialize dataloaders
        if config.dataset_name == 'sdd':
            data_folder = 'data'
            test_dataset = SocialDataset(data_folder, set_name="test", b_size=4096, t_tresh=0, d_tresh=100, scene='sdd')
        elif config.dataset_name == 'eth':
            data_folder = 'data/ETH_UCY'
            test_dataset = SocialDataset(data_folder, set_name="test", b_size=4096, t_tresh=0, d_tresh=50, scene=config.data_scene)

        if config.vis and config.dataset_name == 'eth':
            self.homo_mat = {}
            for scene in ['eth', 'hotel', 'univ', 'zara1', 'zara2']:
                self.homo_mat[scene] = np.loadtxt(f'./data/ETH_image/{scene}_H.txt')

        # Initialize dataloaders
        self.test_dataset = DataLoader(test_dataset, batch_size=1, collate_fn=socialtraj_collate)
        print('Loaded data!')

        if torch.cuda.is_available(): torch.cuda.set_device(config.gpu)

        self.settings = {
            "train_batch_size": config.train_b_size,
            "test_batch_size": config.test_b_size,
            "use_cuda": config.cuda,
            "dim_feature_tracklet": config.past_len * 2,
            "dim_feature_future": config.future_len * 2,
            "dim_embedding_key": config.dim_embedding_key,
            "past_len": config.past_len,
            "future_len": 12,
        }

        config.mode = 'ALL'
        # model
        self.model_Pretrain = torch.load(config.model_Pretrain, map_location=torch.device('cpu')).cuda()
        self.model = Final_Model(config, self.model_Pretrain)

        if config.cuda:
            self.model = self.model.cuda()
        self.start_epoch = 0
        self.config = config

        self.device = torch.device('cuda') if config.cuda else torch.device('cpu')

        


    def print_model_param(self, model):
        total_num = sum(p.numel() for p in model.parameters())
        trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("\033[1;31;40mTrainable/Total: {}/{}\033[0m".format(trainable_num, total_num))
        return 0
    
    
    def fit(self):
        
        dict_metrics_test = self.evaluate(self.test_dataset)
        print('Test FDE_48s: {} ------ Test ADE: {}'.format(dict_metrics_test['fde_48s'], dict_metrics_test['ade_48s']))
        print('-'*100)
        

    def evaluate(self, dataset):
        
        ade_48s = fde_48s = 0
        samples = 0
        dict_metrics = {}
        self.model.eval()

        past_gt = []
        fut_gt = []
        fut_pred_20 = []
        fut_pred_best = []

        with torch.no_grad():
            for _, (trajectory, mask, initial_pos, seq_start_end) in enumerate(dataset):
                trajectory, mask, initial_pos = torch.FloatTensor(trajectory).to(self.device), torch.FloatTensor(mask).to(self.device), torch.FloatTensor(initial_pos).to(self.device)

                traj_norm = trajectory - trajectory[:, self.config.past_len - 1:self.config.past_len, :]
                x = traj_norm[:, :self.config.past_len, :]
                destination = traj_norm[:, -1:, :]
                y = traj_norm[:, self.config.past_len:, :]
                gt = traj_norm[:, 1:self.config.past_len + 1, :]

                abs_past = trajectory[:, :self.config.past_len, :]
                initial_pose = trajectory[:, self.config.past_len - 1:self.config.past_len, :]
                
                output = self.model(x, abs_past, seq_start_end, initial_pose)
                output = output.data

                future_rep = traj_norm[:, 8:, :].unsqueeze(1).repeat(1, 20, 1, 1)
                distances = torch.norm(output - future_rep, dim=3)

                fde_mean_distances = torch.mean(distances[:, :, -1:], dim=2)  # find the tarjectory according to the last frame's distance
                fde_index_min = torch.argmin(fde_mean_distances, dim=1)
                fde_min_distances = distances[torch.arange(0, len(fde_index_min)), fde_index_min]
                fde_48s += torch.sum(fde_min_distances[:, -1])

                ade_mean_distances = torch.mean(distances[:, :, :], dim=2)  # find the tarjectory according to the last frame's distance
                ade_index_min = torch.argmin(ade_mean_distances, dim=1)
                ade_min_distances = distances[torch.arange(0, len(ade_index_min)), ade_index_min]
                ade_48s += torch.sum(torch.mean(ade_min_distances, dim=1))

                samples += distances.shape[0]

                if self.config.vis:
                    if self.config.dataset_name == 'eth':
                        # Transform trajectories from world coordinate system to pixel/image space
                        trajectory_ = np.concatenate((trajectory.cpu().numpy(), np.ones((trajectory.size(0), trajectory.size(1), 1))), -1)
                        trajectory_ = np.matmul(np.linalg.inv(self.homo_mat[self.config.data_scene]), trajectory_.transpose(0, 2, 1)).transpose(0, 2, 1)
                        trajectory_ /= trajectory_[:, :, -1:]

                        output_ = output + initial_pose.unsqueeze(1)
                        output_ = np.concatenate((output_.cpu().numpy(), np.ones((output_.size(0), output_.size(1), output_.size(2), 1))), -1)
                        output_ = np.matmul(np.linalg.inv(self.homo_mat[self.config.data_scene]), output_.transpose(0, 1, 3, 2)).transpose(0, 1, 3, 2)
                        output_ /= output_[:, :, :, -1:]

                        if self.config.data_scene == 'eth' or self.config.data_scene == 'ucy':
                            trajectory_ = np.concatenate((trajectory_[:, :, 1:2], trajectory_[:, :, :1]), -1)
                            output_ = np.concatenate((output_[:, :, :, 1:2], output_[:, :, :, :1]), -1)
                        else:
                            trajectory_ = trajectory_[:, :, :2]
                            output_ = output_[:, :, :, :2]

                    elif self.config.dataset_name == 'sdd':
                        trajectory_ = trajectory.cpu().numpy()
                        output_ = output.cpu().numpy()

                    past_gt.append(trajectory_[:, :self.config.past_len])
                    fut_gt.append(trajectory_[:, self.config.past_len:])
                    fut_pred_20.append(output_)
                    fut_pred_best.append(output_[np.arange(0, len(ade_index_min)), ade_index_min.cpu().numpy()])


            dict_metrics['fde_48s'] = fde_48s / samples
            dict_metrics['ade_48s'] = ade_48s / samples

            if self.config.vis:
                past_gt = np.concatenate(past_gt, 0)
                fut_gt = np.concatenate(fut_gt, 0)
                fut_pred_20 = np.concatenate(fut_pred_20, 0)
                fut_pred_best = np.concatenate(fut_pred_best, 0)

                if self.config.dataset_name == 'eth':
                    vis_ETH(self.config.data_scene, past_gt, fut_gt, fut_pred_20, fut_pred_best)
                # else:
                    # vis_SDD(past_gt, fut_gt, fut_pred_20, fut_pred_best)

        return dict_metrics
