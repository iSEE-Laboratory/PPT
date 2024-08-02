"""
Full definition of a GPT Language Model, all of it in this single file.

References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from models.utils import CfgNode as CN
from einops import repeat
import numpy as np

from models.layer_utils import *
from scipy import interpolate
from copy import deepcopy
# -----------------------------------------------------------------------------

class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.mode = config.mode

    def forward(self, x, mask_type='causal', mask_input=None):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if mask_input != None:
            mask = (mask_input == 0)
            # print(mask_input[0])
        elif mask_type == 'causal':
            mask = (self.bias[:, :, :T, :T] == 0)
        elif mask_type == 'des_causal':
            self.bias[:, :, :, T-1] = 1
            mask = (self.bias[:,:,:T,:T] == 0)
        elif mask_type == 'all':
            self.bias[:, :, :T, :T] = 1
            mask = (self.bias[:, :, :T, :T] == 0)
        elif mask_type == 'pred_3points':
            self.bias[:, :, :T, :T] = 1
            self.bias[:, :, 9, 11] = 0
            self.bias[:, :, 11, 9] = 0
            mask = (self.bias[:, :, :T, :T] == 0)
        elif mask_type == 'pred_11points':
            self.bias[:, :, :T, :T] = 1
            self.bias[:, :, 9:T - 1, 9:T - 1] = 0

            self.bias[:, :, :, 11::4] = 1
            self.bias[:, :, range(9, 20), range(9, 20)] = 1

            mask = (self.bias[:, :, :T, :T] == 0)
        else:
            self.bias[:, :, :T, :T] = 1
            mask = (self.bias[:, :, :T, :T] == 0)

        att = att.masked_fill(mask, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y



class Block(nn.Module):
    """ an unassuming Transformer block """
    def __init__(self, config, pretrain=True):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),  # nice
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x, mask_type='causal', mask_input=None):
        # TODO: check that training still works
        x = x + self.attn(self.ln1(x), mask_type, mask_input)
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    """ GPT Language Model """

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.block_size = config.block_size
        params_given = all([config.n_layer is not None, config.n_head is not None, config.n_embd is not None])

        assert params_given
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.embd_pdrop),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        # self.social = Social_Batch_Attention(config)
        # self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # self.lm_head = MLP(config.n_embd, config.vocab_size, (64,))

        # init all weights, and apply a special scaled init to the residual projections, per GPT-2 paper
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters (note we don't count the decoder parameters in lm_head)
        n_params = sum(p.numel() for p in self.transformer.parameters())
        print("number of parameters: %.2fM" % (n_params/1e6,))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    @classmethod
    def from_pretrained(cls, model_hf, config):
        """
        Initialize a pretrained GPT model by copying over the weights
        from a huggingface/transformers checkpoint.
        """

        model = GPT(config)
        sd_hf = model_hf.state_dict()
        model.load_state_dict(sd_hf)

        return model

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def forward(self, input, social=None, targets=None, mask_type='causal', mask_input=None):
        device = input.device
        b, t, d = input.size()
        # assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        if mask_type == 'causal':
            pos = torch.arange(1, t+1, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)
            if t == 9:
                pos[:, -1] = 19
            tok_emb = input
            pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
            x = self.transformer.drop(tok_emb + pos_emb)
        elif mask_type == 'pred_1point':
            pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)
            pos[0, -2] = 14
            pos[0, -1] = 20
            tok_emb = input
            pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
            x = self.transformer.drop(tok_emb + pos_emb)
        elif mask_type == 'pred_3points':
            pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)
            # 修改了一下位置编码
            pos[0, -4] = 11
            pos[0, -3] = 14
            pos[0, -2] = 17
            pos[0, -1] = 20
            tok_emb = input
            pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
            x = self.transformer.drop(tok_emb + pos_emb)
        elif mask_type == 'pred_11points' or mask_type == 'pred_20points':
            pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)
            tok_emb = input
            pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
            x = self.transformer.drop(tok_emb + pos_emb)
        else:
            pos = torch.arange(1, t+1, dtype=torch.long, device=device).unsqueeze(0)  # shape (1, t)
            tok_emb = input
            pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (1, t, n_embd)
            # pos_emb[:, -1] = 0
            x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x, mask_type, mask_input)
        output_feat = self.transformer.ln_f(x)
        return output_feat


EPSILON = np.finfo(np.float32).tiny


class Final_Model(nn.Module):

    def __init__(self, config, pretrained_model=None, model_ST=None, model_LT=None):
        super(Final_Model, self).__init__()

        self.name_model = 'PPT_Model'
        self.use_cuda = config.cuda
        self.dim_embedding_key = config.dim_embedding_key
        self.past_len = config.past_len
        self.future_len = config.future_len
        self.mode = config.mode

        assert self.mode in ['Short_term', 'Des_warm', 'Long_term', 'ALL'], 'WRONG MODE!'

        # LAYERS for different modes
        if self.mode == 'Short_term':
            self.traj_encoder_1 = nn.Linear(config.vocab_size, config.n_embd)
            self.AR_Model = GPT(config)
            self.predictor_1 = nn.Linear(config.n_embd, config.vocab_size)
        else:
            if self.mode == 'Des_warm':
                self.Traj_encoder = pretrained_model.traj_encoder_1
                self.AR_Model = pretrained_model.AR_Model
                for p in self.parameters():
                    p.requires_grad = False
                self.predictor_Des = MLP(config.n_embd, 20*2, (512, 512, 512))
                self.rand_token = nn.Parameter(torch.rand(1, 1, config.n_embd))
                self.des_encoder = nn.Linear(config.n_embd, config.n_embd)
            else:
                self.Traj_encoder = deepcopy(pretrained_model.Traj_encoder)
                self.AR_Model = deepcopy(pretrained_model.AR_Model)
                self.predictor_Des = deepcopy(pretrained_model.predictor_Des)
                self.rand_token = deepcopy(pretrained_model.rand_token)
                self.des_encoder = deepcopy(pretrained_model.des_encoder)
                if self.mode == 'Long_term':
                    for p in self.parameters():
                        p.requires_grad = True
                elif self.mode == 'ALL':
                    self.traj_trans_layer = deepcopy(pretrained_model.AR_Model)
                    self.traj_encoder = deepcopy(pretrained_model.Traj_encoder)
                    # self.traj_rand_fut_token = deepcopy(pretrained_model.rand_token)
                    # self.traj_rand_fut_token = repeat(deepcopy(pretrained_model.rand_token), 'b () d -> b n d', n=11)
                    self.traj_rand_fut_token = nn.Parameter(deepcopy(pretrained_model.rand_token.data.repeat(1, 11, 1)))
                    self.traj_fut_token_encoder = deepcopy(pretrained_model.des_encoder)
                    self.traj_decoder = nn.Linear(config.n_embd, config.vocab_size)
                    self.traj_decoder_9 = nn.Linear(config.n_embd, config.vocab_size)
                    self.traj_decoder_20 = MLP(config.n_embd, config.vocab_size, (512, 512, 512))
                    self.traj_transform_traj = nn.Linear(config.n_embd, config.n_embd)
                    self.traj_transform_des = nn.Linear(config.n_embd, config.n_embd)

                    # self.traj_trans_layer = GPT(config)
                    # self.traj_encoder = nn.Linear(config.vocab_size, config.n_embd)
                    # self.traj_rand_fut_token = nn.Parameter(torch.rand(1, 11, config.n_embd))
                    # self.traj_fut_token_encoder = nn.Linear(config.n_embd, config.n_embd)
                    # self.traj_decoder_9 = nn.Linear(config.n_embd, config.vocab_size)
                    # self.traj_decoder = nn.Linear(config.n_embd, config.vocab_size)
                    # self.traj_decoder_20 = MLP(config.n_embd, config.vocab_size, (512, 512, 512))


    def get_trajectory(self, past, abs_past, seq_start_end, end_pose):
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


    def forward(self, past, abs_past, seq_start_end, end_pose, destination=None, epoch=None):
        if self.mode == 'Short_term':
            past_state = self.traj_encoder_1(past)
            feat = self.AR_Model(past_state, mask_type='causal')
            pred = self.predictor_1(feat)
            return pred
        elif self.mode == 'Des_warm' or self.mode == 'Long_term':
            past_state = self.Traj_encoder(past)
            des_token = repeat(self.rand_token, '() n d -> b n d', b=past.size(0))
            des_state = self.des_encoder(des_token)
            traj_state = torch.cat((past_state, des_state), dim=1)
            feat = self.AR_Model(traj_state)
            pred_des = self.predictor_Des(feat[:, -1])
            return pred_des.view(pred_des.size(0), 20, -1)
        elif self.mode == 'ALL':
            past_state = self.Traj_encoder(past)
            des_token = repeat(self.rand_token, '() n d -> b n d', b=past.size(0))
            des_state = self.des_encoder(des_token)
            traj_state = torch.cat((past_state, des_state), dim=1)
            feat = self.AR_Model(traj_state)
            pred_des = self.predictor_Des(feat[:, -1])
            pred_des = pred_des.view(pred_des.size(0), 20, -1)
            distances = torch.norm(destination - pred_des, dim=2)
            index_min = torch.argmin(distances, dim=1)
            min_des_traj = pred_des[torch.arange(0, len(index_min)), index_min]
            destination_prediction = min_des_traj
            fut_token = repeat(self.traj_rand_fut_token, '() n d -> b n d', b=past.size(0))

            past_feat = self.traj_encoder(past)
            fut_feat = self.traj_fut_token_encoder(fut_token)
            des_feat = self.traj_encoder(destination_prediction)
            traj_feat = torch.cat((past_feat, fut_feat, des_feat.unsqueeze(1)), 1)
            prediction_feat = self.traj_trans_layer(traj_feat, mask_type='all')
            pre_prediction = self.traj_decoder_9(prediction_feat[:, self.past_len - 1:self.past_len])
            mid_prediction = self.traj_decoder(prediction_feat[:, self.past_len:-2])
            des_prediction = self.traj_decoder_20(prediction_feat[:, -2:-1]) + destination_prediction.unsqueeze(1)
            total_prediction = torch.cat((pre_prediction, mid_prediction, des_prediction), 1)

            # for t in range(1, 12):
            #     total_prediction[:, t - 1:t] = total_prediction[:, t - 1:t] + destination_prediction.unsqueeze(1) * t / 12

            transform_traj_feat = self.traj_transform_traj(prediction_feat[:, self.past_len - 1:-1])
            transform_des_feat = self.traj_transform_des(feat[:, -1])

            return total_prediction, transform_des_feat, transform_traj_feat, destination_prediction
