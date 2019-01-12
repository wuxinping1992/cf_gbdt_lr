from pyspark import SparkContext, HiveContext
from pyspark.mllib.recommendation import Rating
from pyspark.sql import Row
import numpy as np



#load rating data

def Rating_info(sc, file_path):
    rating = sc.textFile(file_path).map(lambda x:x.split('\t')).map(lambda x: Rating(x[0],x[1],x[2]))
    return rating

def split_age(age):
    if age <= 20:
        return 0
    elif age <= 45:
        return 1
    elif age <= 60:
        return 2
    else:
        return 3
def transform_gender(gender):
    if gender == 'F':
        return 0
    else:
        return 1

# transform feature
def transfromed_feature(leaf, num_leaf):
    transfrom_feature_matrix = np.zeros([len(leaf), len(leaf[0]) * num_leaf], dtype=np.int64)
    for i in range(len(leaf)):
        temp = np.arange(len(leaf[0])) * num_leaf - 1 + np.array(leaf[i])
        transfrom_feature_matrix[i][temp] += 1
    return transfrom_feature_matrix

#load user info data

def user_info(sc, file_path):
    user_info=sc.textFile(file_path).map(lambda x: x.split('|'))
    user_info_df = sc.parallelize((Row(user_id=data[0], age=float(split_age(int(data[1]))), gender=transform_gender(data[2]))) for data in user_info.collect()).toDF()
    return user_info_df

#load item info data

def item_info(sc, file_path):
    item_info = sc.textFile(file_path).map(lambda line: line.split("|"))
    item_info_df = sc.parallelize((Row(item_id=data[0], movie_title=data[1], release_date=data[2], \
                                       video_release_data=data[3], imdb_url=data[4], unknow=float(data[5]), \
                                       action=float(data[6]), adventure=float(data[7]), animation=float(data[8]), \
                                       childrens=float(data[9]), comedy=float(data[10]), \
                                       crime=float(data[11]), documentary=float(data[12]), drama=float(data[13]),
                                       fantasy=float(data[14]), film_noir=float(data[15]), horror=float(data[5]), \
                                       musical=float(data[16]), mystery=float(data[17]), romance=float(data[18]), \
                                       sci_fi=float(data[19]), thriller=float(data[10]), war=float(data[21]), \
                                       western=float(data[22]))) for data in item_info.collect()).toDF()
    return item_info_df

#load sample


def sample(sc, rating_file_path, user_file_path, item_file_path, k):
    item_info_df = item_info(sc, item_file_path)
    user_info_df = user_info(sc, user_file_path)
    num_item = range(item_info_df.count())
    #pos example
    pos = sc.textFile(rating_file_path).map(lambda x:x.split('\t'))
    pos_sample = sc.parallelize((Row(user_id=data[0], item_id=data[1], label=float(1))) for data in pos.collect()).toDF()
    pos_user_item = [[int(data[0]), int(data[1])] for data in pos_sample.select(['user_id','item_id']).collect()]
    pos_user_item_dict = {}
    neg_sample = []
    print("starting...")
    for data in pos_user_item:
        if data[0] not in pos_user_item_dict.keys():
            pos_user_item_dict[data[0]] = [data[1]]
        else:
            pos_user_item_dict[data[0]].append(data[1])
    for data in pos_user_item:
        i = 0
        while i<k:
            item = np.random.choice(num_item, 1)
            if item[0] in pos_user_item_dict[data[0]]:
                pass
            else:
               neg_sample = [data[0], item[0], float(1)]
               pos_user_item_dict[data[0]].append(item[0])
    neg_sample_df = sc.parallelize((Row(user_id=data[0], item_id=data[1], label=float(data[3]))) for data in neg_sample)
    sample = pos_sample.union(neg_sample_df)
    print("complete...")
    return sample.join(user_info_df, on='user_id', how='left') \
                  .join(item_info_df, on='item_id', how='left')

def extract_feature_label(df):
    feature = ['age', 'gender', 'action', 'adventure', 'animation', 'childrens', 'comedy', \
               'crime', 'documentary', 'drama', 'fantasy', 'film_noir', 'horrormusical', 'mystery', 'romance', \
               'sci_fi', 'thriller', 'unknow', 'war', 'western']
    data_feature = [[float(data[i])] for i in range(feature) for data in df.select(feature).collect()]
    data_label = [float(data) for data in df.select('label').collect()]
    return data_feature, data_label

if __name__=="__main__":
    sc = SparkContext('local', 'data_process')
    sqlcontext = HiveContext(sc)
    sc.setLogLevel("ERROR")
    rating_file_path = "E:/data/ml-100k/u.data"
    user_file_path = "E:/data/ml-100k/u.user"
    item_file_path = "E:/data/ml-100k/u.item"
    k= 5
    data_sample = sample(rating_file_path, user_file_path, item_file_path, k)
    print(data_sample.count())