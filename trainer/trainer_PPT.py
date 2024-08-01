import os
import datetime
import torch
import torch.nn as nn
# from models.model_AIO import model_encdec
from trainer.evaluations import *
from models.model import Final_Model

import logging
from sddloader import *
from models.preprocessing import *
# for visualization
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F
import yaml
from torch.utils.data import DataLoader
from einops import repeat
from openpyxl import Workbook, load_workbook

torch.set_num_threads(5)

class Trainer:
    def __init__(self, config):
        """
        The Trainer class handles the training procedure for training the autoencoder.
        :param config: configuration parameters (see train_ae.py)
        """

        # test folder creating
        self.name_test = str(datetime.datetime.now())[:10]
        self.folder_test = 'training/training_' + config.mode + '/' + self.name_test + '_' + config.info
        if not os.path.exists(self.folder_test):
            os.makedirs(self.folder_test)
        self.folder_test = self.folder_test + '/'

        obs_len = config.past_len
        fut_len = config.future_len
        total_len = obs_len + fut_len

        print('Preprocess data')

        # dataset_name = config.dataset_name.lower()
        # if dataset_name == 'sdd':
        #     image_file_name = 'reference.jpg'
        # elif dataset_name == 'ind':
        #     image_file_name = 'reference.png'
        # elif dataset_name == 'eth':
        #     image_file_name = 'oracle.png'
        # else:
        #     raise ValueError(f'{dataset_name} dataset is not supported')
        #
        # # ETH/UCY specific: Homography matrix is needed to convert pixel to world coordinates
        # if dataset_name == 'eth':
        #     self.homo_mat = {}
        #     for scene in ['eth', 'hotel', 'students001', 'students003', 'uni_examples', 'zara1', 'zara2', 'zara3']:
        #         self.homo_mat[scene] = torch.Tensor(np.loadtxt(f'data/eth_ucy/{scene}_H.txt')).to(config.gpu)
        #     seg_mask = True
        # else:
        #     self.homo_mat = None
        #     seg_mask = False

        # Initialize dataloaders
        if config.dataset_name == 'sdd':
            data_folder = 'data'
            train_dataset = SocialDataset(data_folder, set_name="train", b_size=512, t_tresh=0, d_tresh=100, scene='sdd')
            val_dataset = SocialDataset(data_folder, set_name="test", b_size=4096, t_tresh=0, d_tresh=100, scene='sdd')
        elif config.dataset_name == 'eth':
            data_folder = 'data/ETH_UCY'
            train_dataset = SocialDataset(data_folder, set_name="train", b_size=256, t_tresh=0, d_tresh=50, scene=config.data_scene)
            val_dataset = SocialDataset(data_folder, set_name="test", b_size=256, t_tresh=0, d_tresh=50, scene=config.data_scene)

        # Initialize dataloaders

        self.train_loader = DataLoader(train_dataset, batch_size=1, collate_fn=socialtraj_collate, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=1, collate_fn=socialtraj_collate)
        print('Loaded data!')

        if torch.cuda.is_available(): torch.cuda.set_device(config.gpu)

        self.max_epochs = config.max_epochs

        if config.mode == 'Short_term':
            self.model = Final_Model(config)
        else:
            self.model_Pretrain = torch.load(config.model_Pretrain, map_location=torch.device('cpu')).cuda()
            self.model_LT = torch.load(config.model_LT, map_location=torch.device('cpu')).cuda() if config.model_LT != None else None
            self.model_ST = torch.load(config.model_ST, map_location=torch.device('cpu')).cuda() if config.model_ST != None else None
            self.model = Final_Model(config, self.model_Pretrain, self.model_ST, self.model_LT)

        # optimizer and learning rate
        self.criterionLoss = nn.MSELoss()
        # if config.mode == 'addressor':
        #     config.learning_rate = 1e-6
        if config.mode == 'ALL':
            self.opt = torch.optim.Adam([
                {"params":[param for name, param in self.model.named_parameters() if 'traj_' in name]},
                {"params":[param for name, param in self.model.named_parameters() if 'traj_' not in name], "lr":config.learning_rate_des},],
                lr=config.learning_rate, #默认参数
            )
        else:
            self.opt = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)

        if config.cuda:
            self.criterionLoss = self.criterionLoss.cuda()
            self.model = self.model.cuda()
        self.start_epoch = 0
        self.config = config
        self.device = torch.device('cuda') if config.cuda else torch.device('cpu')

        self.logger = logging.getLogger('test')
        self.logger.setLevel(level=logging.DEBUG)

        formatter = logging.Formatter('%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

        file_handler = logging.FileHandler(os.path.join(self.folder_test, 'train.log'))
        file_handler.setLevel(level=logging.INFO)
        file_handler.setFormatter(formatter)

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        # stream_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)
        # tensorboard writer
        folder_log = 'training/' + config.dataset_name + '/training_' + config.mode
        self.tb_writer = SummaryWriter(os.path.join(folder_log, 'logs', self.name_test + '_' + config.info))


    def print_model_param(self, model):
        total_num = sum(p.numel() for p in model.parameters())
        trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info("\033[1;31;40mTrainable/Total: {}/{}\033[0m".format(trainable_num, total_num))
        return 0

    # def adjust_learning_rate(self, optimizer, epoch):
    #     lr = optimizer.param_groups[0]['lr'] * (0.1 ** (epoch // 100))  # 学习率每个epoch乘以0.1
    #     # lr = opt.lr * (0.1 ** (epoch // opt.step))	#学习率每10个epoch乘以0.1
    #     return lr

    def fit(self):
        self.print_model_param(self.model)
        minValue = 200
        minADE = 2000
        minFDE = 2000
        for epoch in range(self.start_epoch, self.config.max_epochs):
            self.logger.info(' ----- Epoch: {}'.format(epoch))
            loss = self._train_single_epoch(epoch)
            self.logger.info('Loss: {}'.format(loss))
            self.tb_writer.add_scalar('train/loss', loss, epoch)

            if (epoch + 1) % 1 == 0:
                self.model.eval()
                if self.config.mode == 'Short_term':
                    currentValue = evaluate_ST(self.val_loader, self.model, self.config, self.device)
                elif self.config.mode == 'Long_term' or self.config.mode == 'Des_warm':
                    currentValue = evaluate_LT(self.val_loader, self.model, self.config, self.device)
                else:
                    fde_, currentValue = evaluate_trajectory(self.val_loader, self.model, self.config, self.device)
                self.tb_writer.add_scalar('train/val', currentValue, epoch)

                if self.config.mode == 'ALL':
                    if 3 * currentValue + fde_ < minValue:
                        minValue = 3 * currentValue + fde_
                        minFDE = fde_.item()
                        minADE = currentValue.item()
                        self.logger.info('min ADE value: {}'.format(currentValue))
                        self.logger.info('min FDE value: {}'.format(fde_))
                        torch.save(self.model, self.folder_test + 'model_' + self.name_test)
                elif currentValue<minValue:
                    minValue = currentValue
                    self.logger.info('min ADE value: {}'.format(minValue))
                    torch.save(self.model, self.folder_test + 'model_' + self.name_test)


    def AttentionLoss(self, sim, distance):
        dis_mask = nn.MSELoss(reduction='sum')
        threshold_distance = 80
        mask = torch.where(distance>threshold_distance, torch.zeros_like(distance), torch.ones_like(distance))
        label_sim = (threshold_distance - distance) / threshold_distance
        label_sim = torch.where(label_sim<0, torch.zeros_like(label_sim), label_sim)
        loss = dis_mask(sim*mask, label_sim*mask) / (mask.sum()+1e-5)
        return loss

    def joint_loss(self, pred):     # pred:[B, 20, 2]
        loss = 0.0
        for Y in pred:
            dist = F.pdist(Y, 2) ** 2
            loss += (-dist / self.config.d_scale).exp().mean()
        loss /= pred.shape[0]
        return loss

    def recon_loss(self, pred, gt):
        distances = torch.norm(pred - gt, dim=2)
        index_min = torch.argmin(distances, dim=1)
        min_distances = distances[torch.arange(0, len(index_min)), index_min]
        loss_recon = torch.sum(min_distances) / distances.shape[0]
        return loss_recon

    def loss_function(self, pred, gt):  # pred:[B, 20, 2]   gt:[B, 8, 2]
        # joint loss
        JL = self.joint_loss(pred) if self.config.lambda_j > 0 else 0.0
        RECON = self.recon_loss(pred, gt) if self.config.lambda_recon > 0 else 0.0
        # print('JL', JL * self.config.lambda_j, 'RECON', RECON * self.config.lambda_recon)
        loss = JL * self.config.lambda_j + RECON * self.config.lambda_recon
        return loss

    def joint_loss_traj(self, pred):     # pred:[B, 20, 2]
        loss = 0.0
        from soft_dtw_cuda import SoftDTW
        # Create the "criterion" object
        sdtw = SoftDTW(use_cuda=True, gamma=0.1)

        # # Compute the loss value
        # loss = sdtw(x, y)  # Just like any torch.nn.xyzLoss()

        for i in range(pred.size(1)):
            for j in range(pred.size(1)):
                # soft-DTW discrepancy, approaches DTW as gamma -> 0
                dist = sdtw(pred[:, i], pred[:, j])
                # print(res.shape)
                loss += (-dist / self.config.d_scale).exp().mean()
        loss /= pred.shape[0]
        return loss

    def recon_loss_traj(self, pred, gt):     # pred:[B, 20, 20, 2]   gt:[B, 20, 1, 2]
        # pred = pred.view(pred.shape[0], -1, nk, pred.shape[2])
        # diff = pred - gt
        # print(diff.shape)
        distances = torch.mean(torch.norm(pred - gt.unsqueeze(1), dim=3), 2)
        index_min = torch.argmin(distances, dim=1)
        min_distances = distances[torch.arange(0, len(index_min)), index_min]
        # print(min_distances.shape)
        loss_recon = torch.sum(min_distances) / distances.shape[0]
        # dist = diff.pow(2).sum(dim=-1).sum(dim=0)
        # loss_recon = dist.min(dim=1)[0].mean()
        return loss_recon

    def loss_function_traj(self, pred, gt):  # pred:[B, 20, 2]   gt:[B, 8, 2]
        # joint loss
        JL = self.joint_loss_traj(pred) if self.config.lambda_trajj > 0 else 0.0
        RECON = self.recon_loss_traj(pred, gt) if self.config.lambda_trajrecon > 0 else 0.0
        # print('JL', JL * self.config.lambda_j, 'RECON', RECON * self.config.lambda_recon)
        loss = JL * self.config.lambda_trajj + RECON * self.config.lambda_trajrecon
        return loss

    def vector_angle_velocity(self, pred, gt):
        pred_ = pred[:, 1:] - pred[:, :-1]
        gt_ = gt[:, 1:] - gt[:, :-1]
        cos_dis = torch.sum(1 - torch.cosine_similarity(pred_, gt_, dim=-1))
        velocity_dis = torch.sum(torch.abs(torch.norm(pred_, p=2, dim=-1) - torch.norm(gt_, p=2, dim=-1)))
        # print('cos:', cos_dis, 'vel:', velocity_dis)
        return cos_dis + self.config.lambda_vel*velocity_dis

    def LayerLoss(self, pred, y):
        assert len(y) == len(pred)
        loss = 0.0
        for idx in range(len(y)):
            loss += self.criterionLoss(y[idx], pred[idx])
        return loss

    def L2_Loss(self, pred, gt):     # pred:[B, nf, 2]   gt:[B, nf, 2]
        distances = torch.norm(pred - gt, dim=2)
        loss = torch.sum(torch.mean(distances, dim=1))
        return loss / distances.shape[0]

    def L2_Loss_i(self, pred, gt):     # pred:[B, nf, 2]   gt:[B, nf, 2]
        distances = torch.norm(pred - gt, dim=2)
        loss = torch.mean(distances, dim=1)
        return loss

    def criterionLoss_i(self, pred, gt):     # pred:[B, nf, 2]   gt:[B, nf, 2]
        distances = torch.sum(torch.sum((pred - gt) * (pred - gt), -1), -1)
        loss = distances / gt.size(-1) / gt.size(-2)
        return loss

    def traj_constraint(self, traj):     # pred:[B, nf, 2]   gt:[B, nf, 2]
        vel = torch.norm(traj[:, 1:] - traj[:, :-1], dim=2)
        loss = torch.sum(torch.abs(vel[:, 1:] - vel[:, :-1]))
        return loss / traj.shape[0]

    def _train_single_epoch(self, epoch):
        self.model.train()
        ade_48s = 0
        samples = 0
        count = 0
        train_loss = 0.0
        for _, (trajectory, mask, initial_pos, seq_start_end) in enumerate(self.train_loader):
            trajectory, mask, initial_pos = torch.FloatTensor(trajectory).to(self.device), torch.FloatTensor(mask).to(self.device), torch.FloatTensor(initial_pos).to(self.device)

            traj_norm = trajectory - trajectory[:, self.config.past_len-1:self.config.past_len, :]
            x = traj_norm[:, :self.config.past_len, :]
            destination = traj_norm[:, -1:, :]
            y = traj_norm[:, self.config.past_len:, :]
            gt = traj_norm[:, 1:self.config.past_len+1, :]

            abs_past = trajectory[:, :self.config.past_len, :]
            initial_pose = trajectory[:, self.config.past_len-1, :]

            self.opt.zero_grad()

            if self.config.mode == 'Short_term':
                pred = self.model(traj_norm[:, :-1], abs_past, seq_start_end, initial_pose)
                loss = self.criterionLoss(pred, traj_norm[:, 1:])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0, norm_type=2)
                self.opt.step()
            elif self.config.mode == 'Des_warm':
                pred_des = self.model(x, abs_past, seq_start_end, initial_pose)
                # print(self.criterionLoss(pred, gt), self.loss_function(pred_des, destination))
                # print(self.criterionLoss(pred, gt), self.criterionLoss(pred_des, destination))
                loss = self.loss_function(pred_des, destination)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0, norm_type=2)
                self.opt.step()
                # print(pred_des.requires_grad, destination.requires_grad)
            elif self.config.mode == 'Long_term':
                pred_des = self.model(x, abs_past, seq_start_end, initial_pose)
                # loss = self.criterionLoss(pred, gt) + self.loss_function(pred_des, destination)
                loss = self.loss_function(pred_des, destination)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0, norm_type=2)
                self.opt.step()
            elif self.config.mode == 'ALL':
                traj_state = self.model_ST.traj_encoder_1(traj_norm[:, :-1])
                traj_feat = self.model_ST.AR_Model(traj_state, mask_type='causal')
                teacher_traj = self.model_ST.predictor_1(traj_feat)

                past_state = self.model_LT.Traj_encoder(x)
                des_token = repeat(self.model_LT.rand_token, '() n d -> b n d', b=x.size(0))
                des_state = self.model_LT.des_encoder(des_token)
                soft_feat = self.model_LT.AR_Model(torch.cat((past_state, des_state), dim=1))

                pred_results, des_pred_feat, traj_pred_feat, pred_des = self.model(x, abs_past, seq_start_end, initial_pose, destination, epoch)

                loss = 0.0
                pred = torch.cat((x[:, -1:], pred_results), 1)
                gt_seq = torch.cat((x[:, -1:], y), 1)
                loss += self.L2_Loss(pred_results, y)
                loss += self.config.lambda_desloss * self.L2_Loss(pred_des.unsqueeze(1), y[:, -1:])
                loss += self.config.traj_lambda_soft * self.criterionLoss(traj_pred_feat, traj_feat[:, self.config.past_len - 1:])  # soft loss
                loss += self.config.des_lambda_soft * self.criterionLoss(des_pred_feat, soft_feat[:, -1])

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0, norm_type=2)
                self.opt.step()

            train_loss += loss.item()
            count += 1
        return train_loss / count
