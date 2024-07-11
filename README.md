<p align="center">
<!-- <h1 align="center">Progressive Pretext Task Learning for Human Trajectory Prediction</h1> -->
<strong>Progressive Pretext Task Learning for Human Trajectory Prediction</strong></h1>
  <p align="center">
    <a href='https://xiaotong-lin.github.io/' target='_blank'>Xiaotong Lin</a>&emsp;
    <a href='https://tmliang.github.io/' target='_blank'>Tianming Liang</a>&emsp;
    <a href='https://scholar.google.com/citations?user=w3GjGqoAAAAJ' target='_blank'>Jianhuang Lai</a>&emsp;
    <a href='https://www.isee-ai.cn/~hujianfang/' target='_blank'>Jian-Fang Hu*</a>&emsp;
    <br>
    Sun Yat-sen University
    <br>
    ECCV 2024
  </p>
</p>

</p>
<p align="center">
  <a href='https://arxiv.org/abs/2308.16905'>
    <img src='https://img.shields.io/badge/Arxiv-2308.16905-A42C25?style=flat&logo=arXiv&logoColor=A42C25'>
  </a>
  <a href=''>
    <img src='https://img.shields.io/badge/Paper-PDF-yellow?style=flat&logo=arXiv&logoColor=yellow'>
  </a>

  <a href='https://github.com/iSEE-Laboratory/PPT'>
    <img src='https://img.shields.io/badge/GitHub-Code-black?style=flat&logo=github&logoColor=white'></a>

  </a>
</p>

## 🏠 About
<div style="text-align: center;">
    <img src="assets/teaser.png" width=100% >
</div>
Human trajectory prediction is a practical task of predicting the future positions of pedestrians on the road, which typically covers all temporal ranges from short-term to long-term within a trajectory. However, existing works attempt to address the entire trajectory prediction with a singular, uniform training paradigm, neglecting the distinction between short-term and long-term dynamics in human trajectories. To overcome this limitation, we introduce a novel Progressive Pretext Task learning (PPT) framework, which progressively enhances the model's capacity of capturing short-term dynamics and long-term dependencies for the final entire trajectory prediction. Specifically, we elaborately design three stages of training tasks in the PPT framework. In the first stage, the model learns to comprehend the short-term dynamics through a stepwise next-position prediction task. 
In the second stage, the model is further enhanced to understand long-term dependencies through a destination prediction task. 
In the final stage, the model aims to address the entire future trajectory task by taking full advantage of the knowledge from previous stages. To alleviate the knowledge forgetting, we further apply a cross-task knowledge distillation. Additionally, we design a Transformer-based trajectory predictor, which is able to achieve highly efficient two-step reasoning by integrating a destination-driven prediction strategy and a group of learnable prompt embeddings. We conduct extensive experiments on four popular benchmarks, and the results demonstrate that our approach achieves state-of-the-art performance with high efficiency.
</br>

## 📖 Implementation


## 🔥 News
- [2023-10-27] Release training and evaluation codes, as well as our checkpoints. Let's play with it!
- [2023-09-16] Release a [demo video](https://sirui-xu.github.io/InterDiff/images/teaser_1080p.mp4) 📹.
- [2023-09-01] Our paper is available on the [Arxiv](https://arxiv.org/pdf/2308.16905) &#x1F389; Code/Models are coming soon. Please stay tuned! ☕️

## 📝 TODO List
- [x] Release more demos.
- [x] Data preparation.
- [x] Release training and evaluation (short-term) codes.
- [x] Release checkpoints.
- [ ] Release evaluation (long-term) and optimization codes.
- [ ] Release code for visualization.

## 🔍 Overview

### 💡 Key Insight
<p align="center">
  <img src="assets/5.gif" width="50%"/><img src="assets/24.gif" width="50%"/>
  <img src="assets/50.gif" width="50%"/><img src="assets/38.gif" width="50%"/>
</p>
As shown, we propose a Progressive Pretext Task learning (PPT) framework for trajectory prediction, aiming to incrementally enhance the model's capacity to understand the past trajectory and predict the future trajectory.
Specifically, our framework consists of three stages of progressive training tasks, as illustrated in subfigure (b). In Stage I, we pretrain our predictor on pretext Task-I, aiming to fully understand the short-term dynamics of each trajectory, by predicting the next position of a trajectory of arbitrary length. In Stage II, we further train the predictor on pretext Task-II, intending to capture the long-term dependencies, by predicting the destination of a trajectory.
Once Task-I and Task-II are completed, the model is capable of capturing both the short-term dynamics and long-term dependencies within the trajectory. Finally, in Stage III, we duplicate our model to obtain two predictors: one for destination prediction and another for intermediate prediction. In this stage, we perform Task-III that enables the model to achieve the complete pedestrian trajectory prediction.
For the sake of stable training, we further employ a cross-task knowledge distillation to avoid knowledge forgetting.
<!-- 
### 🧪 Experimental Results

#### Qualitative Comparisons with Pure Diffusion
<p align="center">
  <img src="assets/results.png" align="center" width="100%">
</p> -->


## 🔗 Citation

If you find our work helpful, please cite:

```bibtex
@inproceedings{
   lin2024ppt,
   title={Progressive Pretext Task Learning for Human Trajectory Prediction},
   author={Lin, Xiaotong and Liang, Tianming and Lai, Jianhuang and Hu, Jian-Fang},
   booktitle={ECCV},
   year={2024},
}
```
