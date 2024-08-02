# Stage-I: Train the model on Task-I
python train_PPT.py --info try_ST --gpu 0 --mode Short_term --learning_rate 0.001 --n_layer 3 --max_epochs 3000

# Stage-II (warm-up): Train the model on Task-II
python train_PPT.py --info try_Des_warm --gpu 0 --mode Des_warm --model_Pretrain ./training/Pretrained_Models/SDD/model_ST --learning_rate 0.001 --lambda_j 100

# Stage-II (finetune): Train the model on Task-II
python train_PPT.py --info try_LT --gpu 0 --mode Long_term --model_Pretrain ./training/Pretrained_Models/SDD/model_Des_warm --learning_rate 0.0001 --lambda_j 100

# Stage-III: Train the modelon Task-III
python train_PPT.py --info try_ALL --gpu 0 --mode ALL --model_Pretrain ./training/Pretrained_Models/SDD/model_LT --model_LT ./training/Pretrained_Models/SDD/model_LT --model_ST ./training/Pretrained_Models/SDD/model_ST --learning_rate 0.0015 --traj_lambda_soft 5 --des_lambda_soft 0.5 --lambda_desloss 1 --learning_rate_des 1e-7
