# cf_gbdt_lr
简单的实现推荐系统的召回模型和排序模型，其中召回模型使用协同过滤算法，排序模型使用gbdt+lr算法

使用的数据为ml-100k的数据，
data_process.py 为数据处理脚本
gbdt_lr.model 为gbdt和lr的排序模型的训练脚本
cf_gbdt_lr_prdict.py 为融合ALS和gbdt_lr整体的预测，其中ALS为召回模型