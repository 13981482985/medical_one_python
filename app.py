from flask import Flask, request,session
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import pymysql
from flask_cors import CORS
import creat_table
import knn
import random_forest
import psycopg2
import json
from scipy.stats import ttest_ind
from scipy.stats import ranksums
from scipy.stats import f_oneway
from gevent import pywsgi
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.stats.anova import anova_lm
from sklearn.cluster import KMeans

app = Flask(__name__)

CORS(app)
host = "10.16.48.219"
port = "5432"
database = "software1"
user = "pg"
password = 111111
def connect_pg():
    pg_connection = psycopg2.connect(database=database,
                     user=user,
                     password=password,
                     host=host,
                     port=port)
    return pg_connection


def connect_mysql():
    connection = pymysql.connect(
        host='10.16.48.219',
        user='root',
        password='111111',
        database='medical',
        cursorclass=pymysql.cursors.DictCursor
    )
    return connection


def get_data(connection, table_name):
    query = f"select * from \"{table_name}\""
    data = pd.read_sql(query, connection)
    connection.close()
    return data


# 通过路径  /pca/yangxing
@app.route('/pca', methods=['POST'])
def pca():
    connection = connect_pg()
    # 获取需要降维的特征
    params = request.get_json()
    table_name = params['tableName']
    feature_cols = params['runParams']  # 需要降维的特征列
    # 查询数据库获取连接
    data = get_data(connection, table_name)
    features = data[feature_cols]
    # 数据预处理：处理缺失值
    features = features.fillna(0)
    print(features)
    # 特征标准化
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    # 计算协方差矩阵
    covariance_matrix = np.cov(scaled_features.T)
    # 计算特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    # 选择主成分数量
    # 这里假设选择保留前10个主成分
    n_components = len(feature_cols) - 1
    # 降维
    pca = PCA(n_components=n_components)
    reduced_features = pca.fit_transform(scaled_features)
    result = pd.DataFrame(reduced_features)
    explained_variance_ratio = pca.explained_variance_ratio_
    contribution_list = explained_variance_ratio.tolist()
    return [result.to_dict(), contribution_list]


@app.route("/knn", methods=['POST'])
def get_knn():
    param = request.get_json()
    print(param)
    return knn.knn(param)


@app.route("/featureCreate", methods=['POST'])
def feature_create():
    param = request.get_json()
    new_table = creat_table.create_table(param)
    # 连接到数据库
    connection = connect_mysql()

    # 创建游标对象
    cursor = connection.cursor()

    # 从数据库中读取数据
    query = f"SELECT * FROM {new_table} limit 15"
    cursor.execute(query)

    # 获取查询结果
    rows = cursor.fetchall()

    return rows


@app.route("/randomForest", methods=['POST'])
def fe():
    param = request.get_json()
    # {'tableName1': 'cardio_train', 'tableName2': 'stroke', 'aiName': 'randomForest', 'runParams': ['Case_ID', 'AGE', 'SEX']}
    return random_forest.random_forest(param)


@app.route("/randomForest1", methods=['POST'])
def fe1():
    param = request.get_json()
    # {'tableName1': 'cardio_train', 'tableName2': 'stroke', 'aiName': 'randomForest', 'runParams': ['Case_ID', 'AGE', 'SEX']}
    return random_forest.random_forest1(param)



#   参数 字典值： tabelName:xxx,  aliName: xxx runParams: []
@app.route("/notDiscreteFeatureDesc", methods=['POST'])
def notDiscreteFeatureDesc():
    # 连接到数据库
    param = request.get_json()
    colName = param['runParams'][0]
    pg_connection = connect_pg()
    tableName = param['tableName']
    sql = f"select \"{colName}\" from \"{tableName}\""
    databaseData = pd.read_sql(sql, con=pg_connection);
    databaseData[colName] = pd.to_numeric(databaseData[colName], errors='coerce')
    total_count = databaseData.shape[0]
    print("total_count:");
    # 删除 NaN 值
    databaseData.dropna(subset=[colName], inplace=True)
    # 计算中位数
    median = databaseData[colName].median()
    print("median", median)
    # 计算均值
    mean = databaseData[colName].mean()
    print("mean", mean)
    # 计算众数
    mode = databaseData[colName].mode().iloc[0]
    print("mode", mode)
    # 计算最大值
    max_value = databaseData[colName].max()
    print("max_value", max_value)
    # 计算最小值
    min_value = databaseData[colName].min()
    print("min_value" ,min_value)
    res = {
        "total": total_count,
        "average": str(mean),
        "middle": str(median),
        "min": str(min_value),
        "max": str(max_value),
        "mode": str(mode)

    }
    print("aaaa")
    print(res)
    return json.dumps(res);



#   参数 字典值： tabelName:xxx,  aliName: xxx runParams: []
@app.route("/modePadding", methods=['POST'])
def modePadding():
    # 连接到数据库
    param = request.get_json()
    colName = param['runParams'][0]
    pg_connection = connect_pg()
    tableName = param['tableName']
    sql = f"select \"{colName}\" from {tableName}"
    databaseData = pd.read_sql(sql, con=pg_connection);
    old_data = databaseData.values.tolist()
    # 计算列的众数
    mode_value = databaseData[colName].mode()[0]
    # 对空值进行众数填充
    databaseData[colName] = databaseData[colName].fillna(mode_value)
    res = {
        "old_data": old_data,
        "new_data": databaseData.values.tolist()
    }

    print(res)
    return json.dumps(res);


@app.route("/meanReplacement", methods=['POST'])
def meanReplacement(): # 均数替换
    # 连接到数据库
    param = request.get_json()
    colName = param['runParams'][0]
    pg_connection = connect_pg()
    tableName = param['tableName']
    sql = f"select \"{colName}\" from {tableName}"
    databaseData = pd.read_sql(sql, con=pg_connection);
    # 计算均值
    # databaseData[colName] = pd.to_numeric(databaseData[colName], errors='coerce')
    len = 0;
    sum = 0
    for value in databaseData[colName]:
        len+=1
        number = pd.to_numeric(value)
        if np.isnan(number):
           continue;
        sum+=number;
    mean_value = round(sum/len,2)
    print("均值：", mean_value)
    old_data = databaseData.values.tolist();
    # 替换空值为均值
    databaseData[colName].fillna(str(mean_value), inplace=True)
    # print("替换过后的数据：",databaseData[colName])
    res = {
        "old_data": old_data,
        "new_data": databaseData.values.tolist()
    }
    print("最后的数据为：")
    print(res)
    return json.dumps(res);

@app.route("/nearestNeighborInterpolation", methods=['POST'])
def nearestNeighborInterpolation():
    # 连接到数据库
    param = request.get_json()
    colName = param['runParams'][0]
    pg_connection = connect_pg()
    tableName = param['tableName']
    sql = f"select \"{colName}\" from {tableName}"
    databaseData = pd.read_sql(sql, con=pg_connection);


    old_data = databaseData[colName].values.tolist()

    # 将数据转换为数字类型并进行最邻近插值处理
    databaseData[colName] = pd.to_numeric(databaseData[colName], errors='coerce')
    databaseData[colName] = databaseData[colName].interpolate(method='nearest')

    # 将插值后的数据保存在 new_data 中，并将其转换为字符串
    new_data = databaseData[colName].astype(str).tolist()

    res = {
        "old_data": old_data,
        "new_data": new_data
    }
    print(res)
    return json.dumps(res);
@app.route("/forwardFilling", methods=['POST'])
def forwardFilling():
    # 连接到数据库
    param = request.get_json()
    colName = param['runParams'][0]
    pg_connection = connect_pg()
    tableName = param['tableName']
    sql = f"select \"{colName}\" from {tableName}"
    databaseData = pd.read_sql(sql, con=pg_connection);
    old_data = databaseData[colName].values.tolist()

    # 将缺失值进行前向填充
    databaseData[colName].fillna(method='ffill', inplace=True)

    res = {
        "old_data": old_data,
        "new_data": databaseData[colName].values.tolist()
    }
    print(res)
    return json.dumps(res);

@app.route("/medianReplacement", methods=['POST'])
def medianReplacement():
    # 连接到数据库
    param = request.get_json()
    colName = param['runParams'][0]
    pg_connection = connect_pg()
    tableName = param['tableName']
    sql = f"select \"{colName}\" from {tableName}"
    databaseData = pd.read_sql(sql, con=pg_connection);
    old_data = databaseData[colName].values.tolist()

    # 计算中位数
    median_value = databaseData[colName].median()

    # 使用中位数填充空值
    databaseData[colName].fillna(median_value, inplace=True)

    # 将填充后的数据转换为字符串（如果需要）
    databaseData[colName] = databaseData[colName].astype(str)

    res = {
        "old_data": old_data,
        "new_data": databaseData[colName].values.tolist()
    }
    print(res)
    return json.dumps(res);

@app.route("/regressionAnalysisReplacement", methods=['POST'])
def regressionAnalysisReplacement():
    # 连接到数据库
    param = request.get_json()
    colName = param['runParams'][0]
    pg_connection = connect_pg()
    tableName = param['tableName']
    sql = f"select \"{colName}\" from {tableName}"
    databaseData = pd.read_sql(sql, con=pg_connection);

    old_data = databaseData[colName].values.tolist()

    # 创建一个新的 DataFrame，只包含非空值
    non_null_data = databaseData[databaseData[colName].notnull()]

    # 提取特征和标签
    X = non_null_data.index.values.reshape(-1, 1)
    y = non_null_data[colName].values.reshape(-1, 1)

    # 初始化并拟合线性回归模型
    regression_model = LinearRegression()
    regression_model.fit(X, y)

    # 使用模型进行预测
    X_predict = databaseData.index.values.reshape(-1, 1)
    predicted_values = regression_model.predict(X_predict)
    predicted_values_rounded = np.round(predicted_values, decimals=2)

    # 将预测值填充到原始数据中
    databaseData[colName].fillna(pd.Series(predicted_values_rounded.flatten(), index=databaseData.index), inplace=True)

    # 将填充后的数据转换为字符串
    databaseData[colName] = databaseData[colName].astype(str)

    # 构造返回结果，将填充后的数据转换为列表
    new_data = databaseData[colName].tolist()

    res = {
        "old_data": old_data,
        "new_data": new_data
    }
    print(res)
    return json.dumps(res)


@app.route("/eucarFilling", methods=['POST'])
def eucarFilling():
    # 连接到数据库
    param = request.get_json()
    colName = param['runParams'][0]
    pg_connection = connect_pg()
    tableName = param['tableName']
    sql = f"select * from {tableName}"
    databaseData = pd.read_sql(sql, con=pg_connection);
    old_data = databaseData[colName].values.tolist()

    # 计算中位数
    median_value = databaseData[colName].median()

    # 使用中位数填充空值
    databaseData[colName].fillna(median_value, inplace=True)

    # 将填充后的数据转换为字符串（如果需要）
    databaseData[colName] = databaseData[colName].astype(str)

    res = {
        "old_data": old_data,
        "new_data": databaseData[colName].values.tolist()
    }
    print(res)
    return json.dumps(res)

@app.route("/clusterReplacement", methods=['POST'])
def clusterFilling():
    # 连接到数据库
    param = request.get_json()
    colName = param['runParams'][0]
    pg_connection = connect_pg()
    tableName = param['tableName']
    sql = f"select * from \"{tableName}\""
    databaseData = pd.read_sql(sql, con=pg_connection)
    old_data = databaseData[colName].values.tolist()
    print(databaseData)
    print("data")
    old_data = databaseData[colName].values.tolist()
    sql_query = f"""
         SELECT column_name
         FROM information_schema.columns
         WHERE table_name = '{tableName}' AND column_name !='{colName}'
             AND data_type IN ('integer', 'bigint', 'smallint', 'numeric', 'real', 'double precision');
     """
    cols = pd.read_sql(sql_query, con=pg_connection)
    print(cols)
    # 获取指定列的数据
    specified_column_data = cols['column_name']
    # 遍历指定列的数据
    col = []
    for value in specified_column_data:
        # 判断这个列在数据库中的确实度 如果小于20%就保留 大于就删除
        sq = f"SELECT ROUND((COUNT(*) - COUNT(\"{value}\"))::NUMERIC / COUNT(*)*100, 2) AS empty_ratio FROM \"{tableName}\""
        missRateData = pd.read_sql(sq, con=pg_connection)
        print("缺失率：", missRateData['empty_ratio'][0])
        if(missRateData['empty_ratio'][0]<20.0):
            col.append(value)
    if(len(col) > 10):
        col = cols.head(10)
    print("保留的：", col)
    # 设置要聚类的类别数
    n_clusters = 5
    # 使用 K 均值聚类算法对样本进行聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    # 使用 fit_predict() 方法进行聚类并返回每个样本所属的类别
    data = databaseData[col['column_name']]
    print(data)
    data.fillna(data.mean(), inplace=True)
    clusters = kmeans.fit_predict(data)
    databaseData["label"] = clusters
    # 对于每个类别，找到众数并填充缺失值
    for cluster_id in range(n_clusters):
        # 获取属于当前类别的样本
        cluster_samples = databaseData[databaseData['label'] == cluster_id]

        # 排除缺失值后找到该类别中 colName 属性的众数
        mode_value = cluster_samples[colName].mode().iloc[0]

        # 将该类别中缺失值的样本的 colName 属性填充为众数值
        databaseData.loc[(databaseData['label'] == cluster_id) & (databaseData[colName].isnull()), colName] = mode_value
    # 删除添加的聚类信息列
    new_data = databaseData[colName].values.tolist()
    print("old data:", old_data)
    print("new Data:", new_data)
    res = {
        "old_data": old_data,
        "new_data": new_data
    }
    return json.dumps(res)



@app.route("/singleFactorAnalyze", methods=['POST'])
def singleFactorAnalyze():
    # 连接到数据库
    param = request.get_json()
    colNames = param['runParams']
    groupCol = colNames[0]
    observeCol = colNames[1]
    pg_connection = connect_pg()
    tableName = param['tableName']
    sql = f"select \"range\" from \"field_management\" where \"feature_name\" = '{groupCol}'"
    databaseData = pd.read_sql(sql, con=pg_connection)
    print(databaseData)
    if databaseData['range'].size == 0:
        # 查询数据库
        sql2 = f"select distinct(\"{groupCol}\") as \"range\" from \"{tableName}\""
        dis_value = pd.read_sql(sql2, con=pg_connection);
        index_range = dis_value['range']
    else:
        index_range = databaseData['range'][0]  # 获取分类标签
    list  = []
    classInto = []
    for index in index_range:   # 先默认为二分类检验
        tempClassInfo = {}
        tempClassInfo['className'] = index
        sql = f"select \"{observeCol}\" from \"{tableName}\" where \"{groupCol}\" = '{index}'"
        print("sql",sql)
        temp = pd.read_sql(sql, con=pg_connection)[observeCol]
        tempClassInfo['frequent'] = temp.size  # 总个数
        temp_valid = temp.dropna()  # 去除空值
        tempClassInfo['validFrequent'] = temp_valid.size  # 有效个数
        tempClassInfo['missFrequent'] = temp.size-temp_valid.size  # 缺失值个数
        list.append(temp_valid)
        classInto.append(tempClassInfo)
    group1 = [float(x) for x in list[0]]
    group2 = [float(x) for x in list[1]]
    # 执行 Wilcoxon 检验
    w_stat, p_value_w = ranksums(group1, group2)
    w_stat = round(w_stat, 2)
    p_value_w = round(p_value_w, 2)

    # 执行 t 检验
    t_stat, p_value_t = ttest_ind(group1, group2, equal_var=False)
    t_stat = round(t_stat, 2)
    p_value_t = round(p_value_t, 2)
    # 执行矫正 t 检验
    correct_t_stat, correct_p_value_t = ttest_ind(group1, group2, equal_var=True)
    correct_t_stat = round(correct_t_stat, 2)
    correct_p_value_t = round(correct_p_value_t, 2)
    res = {
        "classInfo": classInto,
        "group1": group1,
        "group2": group2,
        "w_stat": w_stat,
        "p_value_w": p_value_w,
        "t_stat": t_stat,
        "p_value_t": p_value_t,
        "correct_t_stat": correct_t_stat,
        "correct_p_value_t": correct_p_value_t
    }
    print(json.dumps(res))
    return json.dumps(res)

@app.route("/consistencyAnalyze", methods=['POST'])
def consistencyAnalyze():
    # 连接到数据库
    param = request.get_json()
    colNames = param['runParams']
    featureName = colNames[0]
    pg_connection = connect_pg()
    tableName = param['tableName']
    sql = f"select * from \"{tableName}\""
    data = pd.read_sql(sql, con=pg_connection)
    try:
        numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
        numeric_data = data.loc[:, numeric_columns]
        # 将能转换为float的列转换为float类型
        for col in numeric_columns:
            # 尝试转换列数据类型，如果转换失败则保持原样
            numeric_data[col] = pd.to_numeric(numeric_data[col], errors='coerce')
        # 空值处理
        data, numeric_columns = handle_missing_values(numeric_data)
        numeric_columns_filtered = [col for col in numeric_columns if col != featureName]
        # 根据组内因素进行分组
        grouped_data = data.groupby(featureName)
        # 分别计算每个组的平均值和总体平均值
        # [numeric_columns_filtered[0] 写死的要观察的列  featureName是要分组的列名
        group_means = grouped_data.mean()[numeric_columns_filtered[0]]
        overall_mean = data[numeric_columns_filtered[0]].mean()
        # 计算组内方差与总体方差的比例，即 ICC(1)
        numerator = sum((group_means - overall_mean) ** 2) / (len(group_means) - 1)
        denominator = sum((data[numeric_columns_filtered[0]] - overall_mean) ** 2) / (len(data) - 1)
        icc1 = numerator / denominator
        # 执行单因素方差分析
        f_statistic_1, p_value_1 = f_oneway(*[group[numeric_columns_filtered[0]] for name, group in grouped_data])
        # 计算自由度
        df1_1 = len(grouped_data) - 1
        df2_1 = len(data) - len(grouped_data)
        # 构建 Mixed Effects Model
        index = numeric_columns_filtered[0]+' ~ 1'
        model = MixedLM.from_formula(index, data, groups=featureName)
        # 拟合模型
        result = model.fit()
        # 提取组间方差和残差方差
        between_group_var = result.cov_re.iloc[0, 0]
        within_group_var = result.scale

        # 计算 ICC(2)
        icc2 = between_group_var / (between_group_var + within_group_var)
        # 执行方差分析
        f_test_result = result.f_test("Intercept = 0")
        # 提取 F 值、df1、df2 和 p 值
        f_statistic_2 = f_test_result.fvalue
        df1_2 = f_test_result.df_num
        df2_2 = f_test_result.df_denom
        p_value_2 = f_test_result.pvalue
        print("p_value_2",p_value_2)
        ICC1 = {
            "method": "ICC1:one-way random-effects model",
            "type": "ICC1",
            "ICC": round(float(icc1), 2),
            "F": round(float(f_statistic_1), 2),
            "df1": int(df1_1),
            "df2": int(df2_1),
            "p": round(float(p_value_1), 2)
        }
        ICC2 = {
            "method": "ICC2:two-way random-effects model",
            "type": "ICC2",
            "ICC": round(float(icc2), 2),
            "F": round(float(f_statistic_2), 2),
            "df1": int(df1_2),
            "df2": int(df2_2),
            "p": round(float(p_value_2), 2)
        }
        res = {
            "code": 200,
            "ICC1": ICC1,
            "ICC2": ICC2
        }
        print(json.dumps(res))
        return json.dumps(res)
    except Exception as e:
        print("异常处理")
        res = {
            "code": 500
        }
        return json.dumps(res)

def aggregate_mean(data):
    return data.mean()
# 定义空值处理逻辑
def handle_missing_values(df, threshold=0.3):
    # 计算每列的空值率
    null_ratio = df.isnull().mean()
    # 标记空值率高于阈值的列
    columns_to_drop = null_ratio[null_ratio > threshold].index.tolist()
    # 删除空值率高的列
    for col in columns_to_drop:
        df.drop(columns=[col], inplace=True)
    # 对剩余的列中的空值进行均值填充
    df.fillna(df.mean(), inplace=True)
    return df, df.columns.tolist()
if __name__ == '__main__':
    server = pywsgi.WSGIServer(('0.0.0.0', 5000), app)
    server.serve_forever()
