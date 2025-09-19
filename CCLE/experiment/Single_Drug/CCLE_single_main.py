# coding: utf-8
import time
import numpy as np
import pandas as pd
# from import_path import *
from sklearn.model_selection import KFold
from CCLE.experiment.myutils import translate_result, roc_auc, dir_path
from CCLE_single_target import mofgcn_single_target


data_dir = dir_path(k=2) + "CCLE/processed_data/"

# 加载细胞系-药物矩阵
cell_drug = pd.read_csv(data_dir + "cell_drug_binary.csv", index_col=0, header=0)
cell_drug = np.array(cell_drug, dtype=np.float32)
cell_sum = np.sum(cell_drug, axis=1)
drug_sum = np.sum(cell_drug, axis=0)

# 加载药物-指纹特征矩阵
drug_feature = pd.read_csv(data_dir + "drug_feature.csv", index_col=0, header=0)
feature_drug = np.array(drug_feature, dtype=np.float32)

# 加载细胞系-基因特征矩阵
gene = pd.read_csv(data_dir + "gene_feature.csv", index_col=0, header=0)
gene = np.array(gene, dtype=np.float32)

# 加载细胞系-cna特征矩阵
cna = pd.read_csv(data_dir + "cna_feature.csv", index_col=0, header=0)
cna = np.array(cna, dtype=np.float32)

# 加载细胞系-mutaion特征矩阵
mutation = pd.read_csv(data_dir + "mutation_feature.csv", index_col=0, header=0)
mutation = np.array(mutation, dtype=np.float32)
mutation = (mutation != 0).astype(np.float32)
# 加载null_mask
null_mask = np.zeros(cell_drug.shape, dtype=np.float32)

k = 5
kfold = KFold(n_splits=k, shuffle=True, random_state=21)

import os
os.makedirs("./pan_result_data", exist_ok=True)  # Add this line to create the directory
os.makedirs("./result_data", exist_ok=True)
file_drug = open("./pan_result_data/single_drug_epoch.txt", "w")

file_drug_time = open("./pan_result_data/each_drug_time.txt", "w")

n_kfolds = 5
import os
os.makedirs("./result_data", exist_ok=True)
total_targets = np.sum(drug_sum >= 10)  # 满足条件的药物数量
total_trainings = total_targets * n_kfolds * k  # 5x5 fold

count = 0
global_start = time.time()
for target_index in np.arange(cell_drug.shape[1]):
    times = []
    epochs = []
    true_data_s = pd.DataFrame()
    predict_data_s = pd.DataFrame()
    target_pos_index = np.where(cell_drug[:, target_index] == 1)[0]
    if drug_sum[target_index] < 10:
        continue
    for folds in range(n_kfolds):
        start = time.time()
        for train, test in kfold.split(target_pos_index):
            train_index = target_pos_index[train]
            test_index = target_pos_index[test]
            epoch, true_data, predict_data = mofgcn_single_target(
                gene=gene, cna=cna, mutation=mutation,
                drug_feature=feature_drug, response_mat=cell_drug,
                null_mask=null_mask, target_index=target_index,
                train_index=train_index, test_index=test_index,
                evaluate_fun=roc_auc, device="cuda:0",
                degree=3, attention_dim=32
            )

            epochs.append(epoch)
            true_data_s = pd.concat([true_data_s, translate_result(true_data)], ignore_index=True)
            predict_data_s = pd.concat([predict_data_s, translate_result(predict_data)], ignore_index=True)
            count += 1
            elapsed = time.time() - global_start
            avg_time_per_round = elapsed / count
            eta = avg_time_per_round * (total_trainings - count)

            print(f"✅ Progress: {count}/{total_trainings} ({(count / total_trainings) * 100:.2f}%) | "
                  f"Elapsed: {elapsed / 60:.2f} min | ETA: {eta / 60:.2f} min")
        end = time.time()
        times.append(end - start)
    file_drug.write(str(target_index) + ":" + str(epochs) + "\n")
    file_drug_time.write(str(target_index) + ":" + str(times) + "\n")
    true_data_s.to_csv("./pan_result_data/drug_" + str(target_index) + "_" + "true_data.csv")
    predict_data_s.to_csv("./pan_result_data/drug_" + str(target_index) + "_" + "predict_data.csv")
file_drug.close()
file_drug_time.close()




import torch
from CCLE.experiment.myutils import (
    roc_auc, f1_score_binary, accuracy_binary, precision_binary, recall_binary, mcc_binary
)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import numpy as np

true_tensor = (torch.tensor(true_data_s.values.flatten(), dtype=torch.float32) > 0.5).float().cuda()
pred_tensor = torch.tensor(predict_data_s.values.flatten(), dtype=torch.float32).cuda()

n_kfold = 5
fold_size = true_tensor.size(0) // n_kfold

# 分类
cls_metrics = {"AUC": [], "ACC": [], "Precision": [], "Recall": [], "F1": [], "MCC": []}
# 回归
reg_metrics = {"RMSE": [], "MAE": [], "R2": [], "PCC": []}

for i in range(n_kfold):
    start = i * fold_size
    end = (i + 1) * fold_size if i < n_kfold - 1 else true_tensor.size(0)

    t = true_tensor[start:end]
    p = pred_tensor[start:end]

    # NaN 过滤
    mask = ~(torch.isnan(t) | torch.isnan(p))
    t = t[mask]
    p = p[mask]
    t_np = t.detach().cpu().numpy()
    p_np = p.detach().cpu().numpy()

    # 分类指标
    auc = roc_auc(t, p)
    f1, threshold = f1_score_binary(t, p)
    acc = accuracy_binary(t, p, threshold)
    precision = precision_binary(t, p, threshold)
    recall = recall_binary(t, p, threshold)
    mcc = mcc_binary(t, p, threshold)

    cls_metrics["AUC"].append(auc)
    cls_metrics["ACC"].append(acc.item())
    cls_metrics["Precision"].append(precision.item())
    cls_metrics["Recall"].append(recall.item())
    cls_metrics["F1"].append(f1.item())
    cls_metrics["MCC"].append(mcc.item())

    # 回归指标
    rmse = mean_squared_error(t_np, p_np, squared=False)
    mae = mean_absolute_error(t_np, p_np)
    r2 = r2_score(t_np, p_np)
    pcc = pearsonr(t_np, p_np)[0]

    reg_metrics["RMSE"].append(rmse)
    reg_metrics["MAE"].append(mae)
    reg_metrics["R2"].append(r2)
    reg_metrics["PCC"].append(pcc)


# 打印分类结果
print("\n=== 5-Fold Classification Evaluation Result ===")
for key in cls_metrics:
    values = np.array(cls_metrics[key])
    print(f"{key}: {values.mean():.4f} ± {values.std():.4f}")

# 打印回归结果
print("\n=== 5-Fold Regression Evaluation Result ===")
for key in reg_metrics:
    values = np.array(reg_metrics[key])
    arrow = "↓" if key in ["RMSE", "MAE"] else "↑"
    print(f"{key} {arrow}: {values.mean():.4f} ± {values.std():.4f}")
