import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
import pymysql


def queryLabel(tabelName1):
    # 在此处添加连接到数据库的代码，并从数据库中提取数据
    # 建立数据库连接
    connection = pymysql.connect(host='10.16.48.219',  # 数据库主机地址
                                 user='root',  # 数据库用户名
                                 password='111111',  # 数据库密码
                                 database='medical')  # 数据库名称
    # 创建游标对象
    cursor = connection.cursor()



    query = f"SELECT field_name, is_demography, is_physiological, is_sociology FROM t_table_manager where table_name = '{tabelName1}' "  # 替换为您的表名和查询语句
    cursor.execute(query)
    # 获取查询结果
    data = cursor.fetchall()

    # 列名
    columns = ['Feature', 'is_demography', 'is_physiological', 'is_sociology']

    # 将元组数据转换为 DataFrame
    df = pd.DataFrame(data, columns=columns)
    return df;


def calculate_feature_weights(target_column,tabelName1):
    # 读取数据集
    data = fetch_data_from_database(tabelName1)



    # 分离特征和目标列
    X = data.drop(target_column, axis=1)

    y = data[target_column]

    # 训练线性回归模型
    model = LinearRegression()
    model.fit(X, y)

    # 提取特征名称和对应的权重值
    feature_names = X.columns
    feature_weights = model.coef_

    # 创建特征名称和权重的DataFrame
    feature_weights_df = pd.DataFrame({'Feature': feature_names, 'Weight': feature_weights})

    # 按权重值降序排列特征
    feature_weights_df = feature_weights_df.sort_values(by='Weight', ascending=False)

    # 添加所属类别

    label = queryLabel(tabelName1)


    # 将两个 DataFrame 按照 'Feature' 列合并
    result_df = pd.merge(feature_weights_df, label, on='Feature', how='left')

    return result_df


def fetch_data_from_database(tabelName1):
    # 在此处添加连接到数据库的代码，并从数据库中提取数据
    # 建立数据库连接
    connection = pymysql.connect(host='10.16.48.219',  # 数据库主机地址
                                 user='root',  # 数据库用户名
                                 password='111111',  # 数据库密码
                                 database='medical')  # 数据库名称
    # 创建游标对象
    cursor = connection.cursor()

    query = f"SELECT * FROM {tabelName1}"  # 替换为您的表名和查询语句
    cursor.execute(query)

    # 获取查询结果
    data = cursor.fetchall()

    column_names = [column[0] for column in cursor.description]

    # 将数据存储在一个DataFrame对象中，并返回它
    data = pd.DataFrame(data, columns=column_names)  # 假设数据存储在名为data的DataFrame对象中
    return data.drop(['id'], axis=1)


def random_forest(param):
    target_column_name = 'label'  # 替换为你的目标列名称
    feature_weights = calculate_feature_weights(target_column_name,param['tableName1'])
    feature_weights2 = calculate_feature_weights(target_column_name,param['tableName2'])

    common_columns = list(feature_weights.columns.intersection(feature_weights2.columns))
    result_df = feature_weights[common_columns]
    result_df['Weight2'] = feature_weights2['Weight']
    df_cleaned = result_df.dropna(subset=['Weight2'])

    return df_cleaned.to_dict(orient='records')


def random_forest1(param):
    target_column_name = 'label'  # 替换为你的目标列名称
    feature_weights = calculate_feature_weights(target_column_name,param['tableName'])

    return feature_weights.to_dict(orient='records')