import numpy as np
import os
import matplotlib.pyplot as plt

# visualize the trajectories predicted by PPT in the ETH/UCY dataset
def vis_ETH(data_scene, past_gt, fut_gt, PPT_20, PPT_best):
    filename = os.path.join("./data/ETH_image", data_scene+".jpg")

    fut_gt = np.concatenate((past_gt[:, -1:], fut_gt), 1)
    past_gt_final_20 = np.expand_dims(past_gt[:, -1:], 1).repeat(20 ,axis=1)
    PPT_20 = np.concatenate((past_gt_final_20, PPT_20), 2)
    PPT_best = np.concatenate((past_gt[:, -1:], PPT_best), 1)

    for traj_id in range(len(past_gt)):
        # read the images
        if not os.path.exists('./vis_results/{}_imgs'.format(data_scene)):
            os.makedirs('./vis_results/{}_imgs'.format(data_scene))

        im = plt.imread(filename)
        fig, ax = plt.subplots()
        im = ax.imshow(im)
        ax.axis('off')

        # visualize the best-of-20 prediction
        ax.plot(PPT_best[traj_id, :, 0], PPT_best[traj_id, :, 1], marker='o', color='#75bbfd', markersize='2.5', linewidth=1.5, markerfacecolor='#2976bb', label='Ours (PPT)')
        ax.plot(PPT_best[traj_id, -1, 0], PPT_best[traj_id, -1, 1], marker='*', markersize='10', color='#75bbfd')

        ax.plot(fut_gt[traj_id, :, 0], fut_gt[traj_id, :, 1], marker='o', color='#fe0002', markersize='2.5', linewidth=1.5, markerfacecolor='#6e1005', markeredgewidth=0.8, label='GT_future')
        ax.plot(fut_gt[traj_id, -1, 0], fut_gt[traj_id, -1, 1], marker='*', color='#fe0002', markersize='10')
        ax.plot(past_gt[traj_id, :, 0], past_gt[traj_id, :, 1], marker='o', color='#056eee', markersize='2.5', linewidth=1.5, markerfacecolor='#1e488f', markeredgewidth=0.8, label='GT_past')

        plt.legend()
        plt.savefig("./vis_results/{}_imgs/{}_id{}_best.jpg".format(data_scene, data_scene, traj_id), bbox_inches='tight')  # 输入地址，并利用format函数修改图片名称
        plt.clf()
        plt.close('all')

        im = plt.imread(filename)
        fig, ax = plt.subplots()
        im = ax.imshow(im)
        ax.axis('off')

        # visualize the 20 trajectories predicted by PPT
        for num in range(20):
            ax.plot(PPT_20[traj_id, num, :, 0], PPT_20[traj_id, num, :, 1], marker='o', color='#75bbfd', markersize='2.5', linewidth=1.5, markerfacecolor='#2976bb')
            ax.plot(PPT_20[traj_id, num, -1, 0], PPT_20[traj_id, num, -1, 1], marker='*', markersize='10', color='#75bbfd')

        ax.plot(fut_gt[traj_id, :, 0], fut_gt[traj_id, :, 1], marker='o', color='#fe0002', markersize='2.5', linewidth=1.5, markerfacecolor='#6e1005', markeredgewidth=0.8)
        ax.plot(fut_gt[traj_id, -1, 0], fut_gt[traj_id, -1, 1], marker='*', color='#fe0002', markersize='10')
        ax.plot(past_gt[traj_id, :, 0], past_gt[traj_id, :, 1], marker='o', color='#056eee', markersize='2.5', linewidth=1.5, markerfacecolor='#1e488f', markeredgewidth=0.8)

        plt.savefig("./vis_results/{}_imgs/{}_id{}_20.jpg".format(data_scene, data_scene, traj_id), bbox_inches='tight')  # 输入地址，并利用format函数修改图片名称
        plt.clf()
        plt.close('all')


# visualize the trajectories predicted by PPT in the SDD dataset
def vis_SDD(past_gt, fut_gt, PPT_20, PPT_best):
    img_name = np.load("./SDD/env_gt.npy", allow_pickle=True)

    fut_gt = np.concatenate((past_gt[:, -1:], fut_gt), 1)
    past_gt_final_20 = np.expand_dims(past_gt[:, -1:], 1).repeat(20 ,axis=1)
    PPT_20 = np.concatenate((past_gt_final_20, PPT_20), 2)
    PPT_best = np.concatenate((past_gt[:, -1:], PPT_best), 1)


    for traj_id in range(len(past_gt)):
        if not os.path.exists('./vis_results/SDD_imgs'):
            os.makedirs('./vis_results/SDD_imgs')

        filename = os.path.join("./data/SDD/env_imgs", img_name[traj_id], "reference.jpg")
        im = plt.imread(filename)

        x_min = int(min(np.concatenate((past_gt[traj_id, :, 0], fut_gt[traj_id, :, 0], PPT_best[traj_id, :, 0]), 0)) - 20)
        x_max = int(max(np.concatenate((past_gt[traj_id, :, 0], fut_gt[traj_id, :, 0], PPT_best[traj_id, :, 0]), 0)) + 20)
        y_min = int(min(np.concatenate((past_gt[traj_id, :, 1], fut_gt[traj_id, :, 1], PPT_best[traj_id, :, 1]), 0)) - 20)
        y_max = int(max(np.concatenate((past_gt[traj_id, :, 1], fut_gt[traj_id, :, 1], PPT_best[traj_id, :, 1]), 0)) + 20)

        h, w, c = im.shape
        box_width = abs(max(x_min, 0) - min(x_max, w))
        box_height = abs(max(y_min, 0) - min(y_max, h))
        if box_height > box_width:
            intv = (box_height - box_width) // 2
            left = max(x_min - intv, 0)
            right = min(x_max + intv, w)
            down = max(y_min, 0)
            up = min(y_max, h)
        else:
            intv = (box_width - box_height) // 2
            left = max(x_min, 0)
            right = min(x_max, w)
            down = max(y_min - intv, 0)
            up = min(y_max + intv, h)

        im = im[down:up, left:right]

        fig, ax = plt.subplots()
        im = ax.imshow(im)
        ax.axis('off')
        PPT_best[traj_id, :, 0] = PPT_best[traj_id, :, 0] - left
        PPT_best[traj_id, :, 1] = PPT_best[traj_id, :, 1] - down
        fut_gt[traj_id, :, 0] = fut_gt[traj_id, :, 0] - left
        fut_gt[traj_id, :, 1] = fut_gt[traj_id, :, 1] - down
        past_gt[traj_id, :, 0] = past_gt[traj_id, :, 0] - left
        past_gt[traj_id, :, 1] = past_gt[traj_id, :, 1] - down

        ax.plot(PPT_best[traj_id, :, 0], PPT_best[traj_id, :, 1], label='best', marker='o', color='#228B22', markersize='4')
        ax.plot(PPT_best[traj_id, -1, 0], PPT_best[traj_id, -1, 1], marker='*', markersize='10', color='#228B22')
        ax.plot(fut_gt[traj_id, :, 0], fut_gt[traj_id, :, 1], label='gt', marker='o', color='r', markersize='4')
        ax.plot(fut_gt[traj_id, -1, 0], fut_gt[traj_id, -1, 1], marker='*', color='r', markersize='10')
        ax.plot(past_gt[traj_id, :, 0], past_gt[traj_id, :, 1], label='gt', marker='o', color='b', markersize='4')

        plt.legend()
        plt.savefig("./vis_results/SDD_imgs/id{}_best.jpg".format(traj_id), bbox_inches='tight')
        plt.clf()
        plt.close('all')


