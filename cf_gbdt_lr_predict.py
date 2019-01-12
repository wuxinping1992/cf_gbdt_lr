from pyspark import SparkContext, HiveContext
from pyspark.ml.recommendation import ALS

import numpy as np
from sklearn.externals import joblib
from data_process import  Rating_info, transfromed_feature, item_info, user_info

def predict(rating_file_path, user_file_path, item_file_path):
    rating = Rating_info(sc, rating_file_path)
    als = ALS(rank=10, maxIter=6)
    alsmodel = als.fit(rating)
    user_item = alsmodel.recommendForAllUsers(10).map(lambda x:x[1]).flatmap(lambda x:x).toDF()
    item_info_df = item_info(sc, item_file_path)
    user_info_df = user_info(sc, user_file_path)
    df = user_item.join(user_info_df, on='user_id', how='left') \
        .join(item_info_df, on='item_id', how='left')
    feature = ['age', 'gender', 'action', 'adventure', 'animation', 'childrens', 'comedy', \
               'crime', 'documentary', 'drama', 'fantasy', 'film_noir', 'horrormusical', 'mystery', 'romance', \
               'sci_fi', 'thriller', 'unknow', 'war', 'western']
    predict_data = [[float(data[i])] for i in range(feature) for data in df.select(feature).collect()]

    print("starting gdbt...")
    gbdt_model = joblib.load('../model/gbdt_model/gbdt.model')
    leaf = gbdt_model.apply(predict_data)[:,:,0].astype(int)
    print("starting transform")
    transform_feature = transfromed_feature(leaf, leaf.max())
    print("starting lr model...")
    lr_model = joblib.load("../model/lr.model")
    y_pred = lr_model.predict(transform_feature)
    print(y_pred[:10])

if __name__=="__main__":
    sc = SparkContext('local', 'predict')
    sqlcontext = HiveContext(sc)
    sc.setLogLevel("ERROR")
    rating_file_path = "E:/data/ml-100k/u.data"
    user_file_path = "E:/data/ml-100k/u.user"
    item_file_path = "E:/data/ml-100k/u.item"
    predict(rating_file_path, user_file_path, item_file_path)
