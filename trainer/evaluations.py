import torch

def evaluate_ST(dataset, model, config, device):
    ade_48s = 0
    samples = 0

    with torch.no_grad():
        for (trajectory, mask, initial_pos, seq_start_end) in dataset:
            trajectory, mask, initial_pos = torch.FloatTensor(trajectory).to(device), torch.FloatTensor(mask).to(device), torch.FloatTensor(initial_pos).to(device)
            # traj (B, T, 2)
            traj_norm = trajectory - trajectory[:, config.past_len - 1:config.past_len, :]
            x = traj_norm[:, :config.past_len, :]
            destination = traj_norm[:, -2:, :]
            y = traj_norm[:, -1:, :]
            gt = traj_norm[:, 1:config.past_len + 1, :]

            initial_pose = trajectory[:, config.past_len-1, :]
            abs_past = trajectory[:, :config.past_len, :]

            output = model(traj_norm[:, :-1], abs_past, seq_start_end, initial_pose)
            output = output.data

            distances = torch.norm(output - traj_norm[:, 1:], dim=2)
            # ade_48s += torch.sum(distances)
            # print(distances.shape)
            ade_48s += torch.sum(torch.mean(distances, dim=1))
            samples += distances.shape[0]

    return ade_48s / samples


def evaluate_LT(dataset, model, config, device):
    fde_48s = 0
    samples = 0

    with torch.no_grad():
        for (trajectory, mask, initial_pos, seq_start_end) in dataset:
            trajectory, mask, initial_pos = torch.FloatTensor(trajectory).to(device), torch.FloatTensor(mask).to(device), torch.FloatTensor(initial_pos).to(device)

            traj_norm = trajectory - trajectory[:, config.past_len - 1:config.past_len, :]
            x = traj_norm[:, :config.past_len, :]
            destination = traj_norm[:, -2:, :]
            des = traj_norm[:, -1:, :]

            initial_pose = trajectory[:, config.past_len-1, :]
            abs_past = trajectory[:, :config.past_len, :]

            output_des = model(x, abs_past, seq_start_end, initial_pose)
            output_des = output_des.data

            # print(output_des.shape, des.shape)
            distances = torch.norm(output_des - des, dim=2)
            index_min = torch.argmin(distances, dim=1)
            min_distances = distances[torch.arange(0, len(index_min)), index_min]
            # ade_48s += torch.sum(distances)
            # print('dist', min_distances.shape)
            fde_48s += torch.sum(min_distances)
            samples += distances.shape[0]

    return fde_48s / samples

def evaluate_trajectory(dataset, model, config, device):
    # for the fulfillment stage or trajectory stage, we should have a fixed past/intention memory bank.
    ade_48s = fde_48s = 0
    samples = 0
    dict_metrics = {}

    with torch.no_grad():
        for (trajectory, mask, initial_pos, seq_start_end) in dataset:
            trajectory, mask, initial_pos = torch.FloatTensor(trajectory).to(device), torch.FloatTensor(mask).to(device), torch.FloatTensor(initial_pos).to(device)

            traj_norm = trajectory - trajectory[:, config.past_len-1:config.past_len, :]
            x = traj_norm[:, :config.past_len, :]
            destination = traj_norm[:, -1:, :]

            initial_pose = trajectory[:, config.past_len-1, :]
            abs_past = trajectory[:, :config.past_len, :]

            output = model.get_trajectory(x, abs_past, seq_start_end, initial_pose)
            output = output.data
            # print(output.shape)

            future_rep = traj_norm[:, 8:, :].unsqueeze(1).repeat(1, 20, 1, 1)
            distances = torch.norm(output - future_rep, dim=3)

            fde_mean_distances = torch.mean(distances[:, :, -1:], dim=2) # find the tarjectory according to the last frame's distance
            fde_index_min = torch.argmin(fde_mean_distances, dim=1)
            fde_min_distances = distances[torch.arange(0, len(fde_index_min)), fde_index_min]
            fde_48s += torch.sum(fde_min_distances[:, -1])

            ade_mean_distances = torch.mean(distances[:, :, :], dim=2) # find the tarjectory according to the last frame's distance
            ade_index_min = torch.argmin(ade_mean_distances, dim=1)
            ade_min_distances = distances[torch.arange(0, len(ade_index_min)), ade_index_min]
            ade_48s += torch.sum(torch.mean(ade_min_distances, dim=1))

            # future_rep = traj_norm[:, 8:, :].unsqueeze(1).repeat(1, 20, 1, 1)
            # distances = torch.norm(output - future_rep, dim=3)
            # mean_distances = torch.mean(distances[:, :, -1:], dim=2)  # find the tarjectory according to the last frame's distance
            # index_min = torch.argmin(mean_distances, dim=1)
            # min_distances = distances[torch.arange(0, len(index_min)), index_min]
            #
            # fde_48s += torch.sum(min_distances[:, -1])
            # ade_48s += torch.sum(torch.mean(min_distances, dim=1))
            samples += distances.shape[0]


        dict_metrics['fde_48s'] = fde_48s / samples
        dict_metrics['ade_48s'] = ade_48s / samples

    return dict_metrics['fde_48s'], dict_metrics['ade_48s']
