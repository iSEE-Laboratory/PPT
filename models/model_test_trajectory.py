import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import repeat
from copy import deepcopy

class Final_Model(nn.Module):
    
    def __init__(self, config, pretrained_model):
        super(Final_Model, self).__init__()

        self.name_model = 'PPT_Model_Test'
        self.use_cuda = config.cuda
        self.dim_embedding_key = 128
        self.past_len = config.past_len
        self.future_len = config.future_len
        self.mode = config.mode

        assert self.mode == 'ALL', 'WRONG MODE!'

        self.Traj_encoder = deepcopy(pretrained_model.Traj_encoder)
        self.AR_Model = deepcopy(pretrained_model.AR_Model)
        self.predictor_Des = deepcopy(pretrained_model.predictor_Des)
        self.rand_token = deepcopy(pretrained_model.rand_token)
        self.des_encoder = deepcopy(pretrained_model.des_encoder)

        self.traj_trans_layer = pretrained_model.traj_trans_layer
        self.traj_encoder = pretrained_model.traj_encoder
        self.traj_rand_fut_token = pretrained_model.traj_rand_fut_token
        self.traj_fut_token_encoder = pretrained_model.traj_fut_token_encoder
        self.traj_decoder = pretrained_model.traj_decoder
        self.traj_decoder_9 = pretrained_model.traj_decoder_9
        self.traj_decoder_20 = pretrained_model.traj_decoder_20

        for p in self.parameters():
            p.requires_grad = False


    def forward(self, past, abs_past, seq_start_end, end_pose):
        predictions = torch.Tensor().cuda()

        past_state = self.Traj_encoder(past)
        des_token = repeat(self.rand_token, '() n d -> b n d', b=past.size(0))
        des_state = self.des_encoder(des_token)
        traj_state = torch.cat((past_state, des_state), dim=1)
        feat = self.AR_Model(traj_state)
        pred_des = self.predictor_Des(feat[:, -1])
        destination_prediction = pred_des.view(pred_des.size(0), 20, -1)

        for i in range(20):
            fut_token = repeat(self.traj_rand_fut_token, '() n d -> b n d', b=past.size(0))

            # ----------- Block 1 --------------
            past_feat = self.traj_encoder(past)
            fut_feat = self.traj_fut_token_encoder(fut_token)
            des_feat = self.traj_encoder(destination_prediction[:, i])
            traj_feat = torch.cat((past_feat, fut_feat, des_feat.unsqueeze(1)), 1)
            prediction_feat = self.traj_trans_layer(traj_feat, mask_type='all')

            pre_prediction = self.traj_decoder_9(prediction_feat[:, self.past_len - 1:self.past_len])
            mid_prediction = self.traj_decoder(prediction_feat[:, self.past_len:-2])
            des_prediction = self.traj_decoder_20(prediction_feat[:, -2:-1]) + destination_prediction[:, i].unsqueeze(1)
            total_prediction = torch.cat((pre_prediction, mid_prediction, des_prediction), 1)
            # for t in range(1, 12):
            #     total_prediction[:, t - 1:t] = total_prediction[:, t - 1:t] + destination_prediction[:, i].unsqueeze(1) * t / 12

            prediction_single = total_prediction
            predictions = torch.cat((predictions, prediction_single.unsqueeze(1)), dim=1)
        return predictions