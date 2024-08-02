import glob
import pickle
import torch
from torch.utils import data
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
from models.preprocessing import *

# Code for this dataloader is heavily borrowed from PECNet.
# https://github.com/HarshayuGirase/Human-Path-Prediction


def initial_pos_func(traj_batches):
	batches = []
	for b in traj_batches:
		starting_pos = b[:, 7, :].copy()/1000 #starting pos is end of past, start of future. scaled down.
		batches.append(starting_pos)

	return batches


class SocialDataset(data.Dataset):

	def __init__(self, folder, scene, set_name="train", b_size=4096, t_tresh=60, d_tresh=50, homo_matrix=None, verbose=False):
		'Initialization'
		load_name = "./{0}/social_{1}_{2}_{3}_{4}_{5}.pickle".format(folder, scene, set_name, b_size, t_tresh, d_tresh)

		# print(load_name)
		with open(load_name, 'rb') as f:
			data = pickle.load(f)

		traj, masks = data
		traj_new = []

		for t in traj:
			t = np.array(t)
			t = t[:, :, 2:4]
			traj_new.append(t)
			if set_name == "train":
				#augment training set with reversed tracklets...
				# reverse_t = np.flip(t, axis=1).copy()
				ks = [1, 2, 3]
				for k in ks:
					data_rot = rot(t, k).copy()
					traj_new.append(data_rot)
				data_flip = fliplr(t).copy()
				traj_new.append(data_flip)

		masks_new = []
		for m in masks:
			masks_new.append(m)

			if set_name == "train":
				#add second time for the reversed tracklets...
				masks_new.append(m)
				for _ in range(3):
					masks_new.append(m)

		seq_start_end_list = []
		for m in masks:
			total_num = m.shape[0]
			scene_start_idx = 0
			num_list = []
			for i in range(total_num):
				if i < scene_start_idx:
					continue
				scene_actor_num = np.sum(m[i])
				scene_start_idx += scene_actor_num
				num_list.append(scene_actor_num)

			cum_start_idx = [0] + np.cumsum(np.array(num_list)).tolist()
			seq_start_end = [(start, end) for start, end in zip(cum_start_idx, cum_start_idx[1:])]
			seq_start_end_list.append(seq_start_end)
			if set_name == "train":
				#add second time for the reversed tracklets...
				seq_start_end_list.append(seq_start_end)
				for _ in range(3):
					seq_start_end_list.append(seq_start_end)

		# print(len(traj_new), len(seq_start_end_list), len(masks_new))
		traj_new = np.array(traj_new)
		masks_new = np.array(masks_new)

		self.trajectory_batches = traj_new.copy()
		self.mask_batches = masks_new.copy()
		self.initial_pos_batches = np.array(initial_pos_func(self.trajectory_batches)) #for relative positioning
		self.seq_start_end_batches = seq_start_end_list
		if verbose:
			print("Initialized social dataloader...")

	def __len__(self):
		return len(self.trajectory_batches)

	def __getitem__(self, idx):
		trajectory = self.trajectory_batches[idx]
		mask = self.mask_batches[idx]
		initial_pos = self.initial_pos_batches[idx]
		seq_start_end = self.seq_start_end_batches[idx]
		return np.array(trajectory), np.array(mask), np.array(initial_pos), np.array(seq_start_end)


def socialtraj_collate(batch):
	trajectories = []
	mask = []
	initial_pos = []
	seq_start_end = []
	for _batch in batch:
		trajectories.append(_batch[0])
		mask.append(_batch[1])
		initial_pos.append(_batch[2])
		seq_start_end.append(_batch[3])
	return torch.Tensor(trajectories).squeeze(0), torch.Tensor(mask).squeeze(0), torch.Tensor(initial_pos).squeeze(0), torch.tensor(seq_start_end ,dtype=torch.int32).squeeze(0)
