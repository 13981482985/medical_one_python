import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import pymysql

from sklearn.decomposition import PCA

# 连接到数据库并提取数据
def fetch_data_from_database(columns_to_keep,tableName):
    # 在此处添加连接到数据库的代码，并从数据库中提取数据
    # 建立数据库连接
    connection = pymysql.connect(host='10.16.48.219',  # 数据库主机地址
                                 user='root',  # 数据库用户名
                                 password='111111',  # 数据库密码
                                 database='medical')  # 数据库名称
    # 创建游标对象
    cursor = connection.cursor()


    # 执行查询语句
    columns_str = ', '.join(columns_to_keep)
    query = f"SELECT {columns_str} FROM {tableName} LIMIT 1000"  # 替换为您的表名和查询语句
    cursor.execute(query)

    # 获取查询结果
    data = cursor.fetchall()

    # 将数据存储在一个DataFrame对象中，并返回它
    data = pd.DataFrame(data)  # 假设数据存储在名为data的DataFrame对象中
    return data

# 数据预处理
def preprocess_data(data):
    # 检查数据是否为空
    if data.empty:
        raise ValueError("Data is empty!")

    # 转换为NumPy数组
    data_array = data.values

    # 需要对特征进行标准化处理
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_array)
    return scaled_data

# 使用KNN算法进行聚类
def perform_knn_clustering(data, n_clusters):
    knn = KMeans(n_clusters=n_clusters)
    knn.fit(data)
    labels = knn.labels_
    return labels

# 执行PCA降维
def perform_pca(data, n_components):
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(data)
    return reduced_data

# 主函数
def knn(param):
    # 选中哪些列进行聚类
    # 指定要进行聚类的列
    tableName = param['tableName']
    columns_to_keep = param['runParams']  # 替换为要聚类的列名

    # 从数据库中获取数据
    data = fetch_data_from_database(columns_to_keep,tableName)

    # 数据预处理
    scaled_data = preprocess_data(data)

    # 执行KNN聚类
    n_clusters = 5  # 设置聚类的簇数
    labels = perform_knn_clustering(scaled_data, n_clusters)

    # 执行PCA降维
    n_components = 2  # 设置降维后的维度
    reduced_data = perform_pca(scaled_data, n_components)

    # 合并数据
    merged_data = np.hstack((reduced_data, labels.reshape(-1, 1)))

    print(merged_data)
    return merged_data.tolist()


