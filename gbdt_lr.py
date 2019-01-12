import numpy as np
from pyspark import SparkContext, HiveContext
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals import joblib
from data_process import sample, extract_feature_label


# gbdt training

def gbdt_train(train_data, train_label):
    print("starting training gbdt model...")
    gbdt = GradientBoostingClassifier(n_estimators=100, max_leaf_nodes=31)
    print("saving gbdt model...")
    joblib.dump(gbdt, '../model/gbdt_model/gbdt.model')
    print("starting prediction leaf...")
    gbdt_model = gbdt.fit(train_data, train_label)
    leaf = gbdt_model.apply(train_data)[:,:,0].astype(int)
    return leaf


#transform feature
def transfromed_feature(leaf, num_leaf):
    transfrom_feature_matrix=np.zeros([len(leaf), len(leaf[0]) * num_leaf], dtype=np.int64)
    for i in range(len(leaf)):
        temp=np.arange(len(leaf[0])) * num_leaf - 1 + np.array(leaf[i])
        transfrom_feature_matrix[i][temp] += 1
    return transfrom_feature_matrix


# lr model
def lr_train(train_data, train_label):
    print('starting training lr model')
    lr = LogisticRegression(penalty='l2')
    lr_model = lr.fit(train_data, train_label)

    print("saving lr model...")
    joblib.dump(lr, '../model/gbdt_model/lr.model')
    # prediction
    print("starting lr model predict...")
    y_pred = lr_model.predict(train_data)
    y_prob = lr_model.predict_proba(train_data)[:,1]

    print("evaluation model")
    pr = float(np.sum([1 if y_pred[i] == train_label[i] else 0 for i in range(len(train_label))]))/float(len(train_label))
    print("prediction precision: " + str(pr))


def train(rating_file_path, user_file_path, item_file_path, k):
    data_sample = sample(sc, rating_file_path, user_file_path, item_file_path, k)
    train_data, train_label = extract_feature_label(data_sample)
    leaf = gbdt_train(train_data, train_label)
    leaf_transform = transfromed_feature(leaf, leaf.max())
    lr_train(leaf_transform, train_label)

if __name__=="__main__":
    sc = SparkContext('local', 'traing')
    sqlcontext = HiveContext(sc)
    sc.setLogLevel("ERROR")
    rating_file_path = "E:/data/ml-100k/u.data"
    user_file_path = "E:/data/ml-100k/u.user"
    item_file_path = "E:/data/ml-100k/u.item"
    k= 5
    train(rating_file_path, user_file_path, item_file_path, k)

