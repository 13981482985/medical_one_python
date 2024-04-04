import pymysql
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
from sqlalchemy import create_engine

# 数据库连接信息
host = '10.16.48.219'
user = 'root'
password = '111111'
database = 'medical'

def fetch_data_from_database():
    # 连接到数据库
    connection = pymysql.connect(host=host, user=user, password=password, database=database)

    # 创建游标对象
    cursor = connection.cursor()

    # 从数据库中读取数据
    query = "SELECT * FROM cardio_train"
    cursor.execute(query)

    # 获取查询结果
    result = cursor.fetchall()

    # 获取列名
    column_names = [column[0] for column in cursor.description]

    # 关闭数据库连接
    connection.close()

    # 将数据存储在一个DataFrame对象中，并返回它
    data = pd.DataFrame(result, columns=column_names)
    return data,column_names

def discretize_data(data,auto_discretize_features):
    # 创建新的DataFrame保存离散化结果
    discretized_data = pd.DataFrame()

    # 保留原来表中的'id'列
    discretized_data['id'] = data['id']

    # 对每一列特征进行离散化
    for feature in auto_discretize_features:
        # 判断特征值个数是否小于等于10
        if data[feature].nunique() >= 10:
            discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
            discretized_feature = discretizer.fit_transform(data[feature].values.reshape(-1, 1))
            discretized_data[feature] = discretized_feature.flatten()
        else:
            discretized_data[feature] = data[feature]


    return discretized_data

def manual_discretize_data(data, bins_dict):
    # 创建新的DataFrame保存离散化结果
    discretized_data = pd.DataFrame()

    # 保留原来表中的'id'列
    discretized_data['id'] = data['id']

    # 对需要手动离散化的特征进行处理
    for feature, bins_ranges in bins_dict.items():
        discretizer = KBinsDiscretizer(n_bins=len(bins_ranges) + 1, encode='ordinal', strategy='uniform')
        discretized_feature = discretizer.fit_transform(data[feature].values.reshape(-1, 1))
        discretized_data[feature] = discretized_feature.flatten()

    return discretized_data

def save_to_database(data):
    # 连接到数据库
    # connection = pymysql.connect(host=host, user=user, password=password, database=database)
    # 创建SQLAlchemy连接
    engine = create_engine(f'mysql+pymysql://{user}:{password}@{host}/{database}')


    # 将离散化结果保存到数据库的新表中
    new_table = 'discretized_table'
    # data.to_sql(new_table, connection, index=False, if_exists='replace')
    data.to_sql(new_table, engine, index=False, if_exists='replace')

    #关闭数据库连接
    # connection.close()
    return new_table

def create_table(param):
    # 从数据库中获取数据
    data, column_names = fetch_data_from_database()

    name_list = []

    for item in param:
        name_list.append(item["name"])


    # 自动离散化的特征列列表

    auto_discretize_features = [x for x in column_names if x not in name_list]
    auto_discretize_features.remove("id")

    # 手动离散化的特征列及对应的划分范围
    range_split = []
    for item in param:
        range_split.append(item["rangeSplit"])

    manual_bins_dict = {}
    for i, name in enumerate(name_list):
        max_values = [item['max'] for item in range_split[i]]
        manual_bins_dict[name] = max_values

    # 进行自动离散化
    auto_discretized_data = discretize_data(data,auto_discretize_features)

    # 进行手动离散化
    manual_discretized_data = manual_discretize_data(data, manual_bins_dict)

    # 将两部分数据合并
    discretized_data = pd.merge(auto_discretized_data, manual_discretized_data, on='id')

    # 获取'cardio'列
    cardio_column = discretized_data['cardio']
    # 从离散化结果中删除'cardio'列
    discretized_data = discretized_data.drop(columns=['cardio'])

    # 将'cardio'列添加到最后一列
    discretized_data['cardio'] = cardio_column

    # 保存离散化结果到数据库的新表
    new_table = save_to_database(discretized_data)
    return new_table





