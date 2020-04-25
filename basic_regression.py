import os
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import SGDRegressor, LinearRegression, ElasticNet, ElasticNetCV
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.tree import DecisionTreeRegressor

####################################################
def regressor_predict(reg, data):
    return reg.predict(data)

def kmeans_knn_regressor_predict(reg, clusters_features, clusters_labels, data, cluster_num):
    cluster_indicies = reg.predict(data)
    predictions = [x for x in range(len(data))]
    for i in range(cluster_num):
        mask = (cluster_indicies == i)
        reg = knn_regressor_build(clusters_features[i], clusters_labels[i])
        for k in range(len(mask)):
            if mask[k]:
                predictions[k] = regressor_predict(reg, [data[k]])[0]
    return predictions

def kmeans_sgd_regressor_predict(reg, clusters_features, clusters_labels, data, cluster_num):
    cluster_indicies = reg.predict(data)
    predictions = [x for x in range(len(data))]
    for i in range(cluster_num):
        mask = (cluster_indicies == i)
        reg = sgd_regressor_build(clusters_features[i], clusters_labels[i])
        for k in range(len(mask)):
            if mask[k]:
                predictions[k] = regressor_predict(reg, [data[k]])[0]
    return predictions

def kmeans_lin_regressor_predict(reg, clusters_features, clusters_labels, data, cluster_num):
    cluster_indicies = reg.predict(data)
    predictions = [x for x in range(len(data))]
    for i in range(cluster_num):
        mask = (cluster_indicies == i)
        reg = lin_regressor_build(clusters_features[i], clusters_labels[i])
        for k in range(len(mask)):
            if mask[k]:
                predictions[k] = regressor_predict(reg, [data[k]])[0]
    return predictions
####################################################
def knn_regressor_build(x_train, y_train):
    param_grid = {
        'weights': ['uniform', 'distance'],
        'n_neighbors': list([min(x, int(0.8 * len(x_train))) for x in (list(range(1, 15)) + list(range(15, 40, 5)))])

    }
    reg_cv = GridSearchCV(KNeighborsRegressor(), param_grid=param_grid, cv=min(5, len(x_train)), n_jobs=-1, iid=False)
    reg_cv.fit(x_train, y_train)
    # print(reg_cv.best_params_, end="\t# ")
    # print(reg_cv.best_score_, end="\t# ")
    return KNeighborsRegressor(n_neighbors=reg_cv.best_params_['n_neighbors'],
                               weights=reg_cv.best_params_['weights']).fit(x_train, y_train)

def kmeans_regressor_build(x_train, y_train):
    cluster_num = 2
    reg = KMeans(n_clusters=cluster_num, n_init=1, algorithm='auto')
    samples_prediction = reg.fit_predict(x_train)
    clusters_features = []
    clusters_labels = []
    for c in range(cluster_num):
        mask = (samples_prediction == c)
        clusters_features.append(x_train[np.where(mask)])
        clusters_labels.append(y_train[np.where(mask)])
    return reg, clusters_features, clusters_labels, cluster_num

def sgd_regressor_build(x_train, y_train):
    param_grid = {
        'loss': ['squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'],
        'penalty': ['elasticnet', 'none', 'l1', 'l2'],
        'fit_intercept': [True, False],
        'learning_rate': ['adaptive']
    }
    reg_cv = GridSearchCV(SGDRegressor(max_iter=2500), param_grid=param_grid, cv=min(5, len(x_train)), n_jobs=-1, iid=False)
    reg_cv.fit(x_train, y_train)
    # print(reg_cv.best_params_, end="\t# ")
    # print(reg_cv.best_score_, end="\t# ")
    return SGDRegressor(loss=reg_cv.best_params_['loss'], max_iter=2500, learning_rate=reg_cv.best_params_['learning_rate'],
                        penalty=reg_cv.best_params_['penalty'], fit_intercept=reg_cv.best_params_['fit_intercept']).fit(x_train, y_train)

def lin_regressor_build(x_train, y_train,):
    param_grid = {
        'fit_intercept': [True, False],
        'normalize': [True, False]
    }

    reg_cv = GridSearchCV(LinearRegression(), param_grid=param_grid, cv=min(5, len(x_train)), n_jobs=-1,  iid=False)
    reg_cv.fit(x_train, y_train)
    # print(reg_cv.best_params_, end="\t# ")
    # print(reg_cv.best_score_, end="\t# ")
    return LinearRegression(fit_intercept=reg_cv.best_params_['fit_intercept'], normalize=reg_cv.best_params_['normalize']).fit(x_train, y_train)

def enet_regressor_build(x_train, y_train):
    # param_grid = {
    #     'fit_intercept': [True, False],
    #     'normalize': [True, False],
    #     # 'alpha': [ 0.5, 0.6, 0.7, 0.7, 0.8, 0.9], #0.1, 0.2, 0.3, 0.4,
    #     # 'l1_ratio': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.7, 0.8, 0.9, 1]
    # }
    reg_cv =  ElasticNetCV(cv=min(5, len(x_train)))
    reg_cv.fit(x_train, y_train)
    return ElasticNet(alpha=reg_cv.alpha_, l1_ratio=reg_cv.l1_ratio, normalize=reg_cv.normalize, fit_intercept=reg_cv.fit_intercept).fit(x_train, y_train)

def dtree_regressor_build(x_train, y_train):
    param_grid = {
        'splitter': ['best', 'random'],
        'criterion': ['mse', 'friedman_mse', 'mae']
    }
    reg_cv = GridSearchCV(DecisionTreeRegressor(), param_grid=param_grid, cv=min(5, len(x_train)), n_jobs=-1,
                          iid=False, scoring='neg_mean_absolute_error')
    reg_cv.fit(x_train, y_train)
    # print(reg_cv.best_params_, end="\t# ")
    # print(reg_cv.best_score_, end="\t# ")
    return DecisionTreeRegressor(splitter=reg_cv.best_params_['splitter'], criterion=reg_cv.best_params_['criterion']).fit(x_train, y_train)
####################################################

def add_to_result(result, differences, type, hours_ahead, data):
    result = pd.concat([result, pd.DataFrame({type + str(hours_ahead - 1): data})], axis=1, sort=False)
    if type != 'real values':
        diff = (result[type + str(hours_ahead - 1)] - result['real values' + str(hours_ahead - 1)]).abs()
        differences = pd.concat([differences, pd.DataFrame({'diff ' + type + str(hours_ahead - 1): diff})], axis=1, sort=False)
    return result, differences


def extract_label_and_features(df, hoursAhead):
    input_cols = [col for col in df if 'AGO' in col]
    labels_cols = str(hours_ahead) + 'AHEAD wavepower'

    features = df[input_cols]
    target = df[labels_cols]
    return features, target
####################################################

def print_results(hours_ahead, type):
    count = differences['diff ' + type + ' predictions' + str(hours_ahead - 1)].count()
    sum = differences['diff ' + type + ' predictions' + str(hours_ahead - 1)].sum()
    mae = sum / count
    mse = np.sum([x*x for x in differences['diff ' + type + ' predictions' + str(hours_ahead - 1)]])/count
    min = differences['diff ' + type + ' predictions' + str(hours_ahead - 1)].min()
    max = differences['diff ' + type + ' predictions' + str(hours_ahead - 1)].max()
    to_print = '{}:\tmse: {:.3f}, mae: {:.3f}, count: {:.3f}, max: {:.3f}, min: {:.3f}, sum: {:.3f}'.format(type,mae,mse,count,max,min,sum)
    print(to_print)
    return (type, mae, mse, hours_ahead)

def make_plot(predictions, y_test, type, best_num, hours_ahead):
    plt.plot(list(range(0, len(y_test))), y_test, label='real results')
    plt.plot(list(range(0, len(predictions))), predictions, label='prediction') #FIXME
    plt.legend()
    mse = mean_squared_error(predictions, y_test)
    mae = mean_absolute_error(predictions, y_test)
    title = type + ' - ' + str(best_num) + ' best features, ' + str(hours_ahead) + ' hours ahead'
    error = "MAE: {:.3f}, MSE: {:.3f}".format(mae,mse)
    plt.title(title + '\n' + error)
    if save_graphs:
        plt.savefig('results/' + title + '.png')
    plt.show()
    mse_csv.at[hours_ahead, type+'-'+str(best_num)] = mse
    mae_csv.at[hours_ahead, type+'-'+str(best_num)] = mae

def create_best(hours_ahead):
    best = pd.read_csv('data/best_features.csv', index_col=0, header=None)
    if force_recalculate_best or not (str(hours_ahead) + 'AHEAD' in best.index):
        # prepare train data
        print("prepare train data", end=', ')
        df_train = pd.read_csv('data/2018_input.csv')
        x_train2, y_train2 = extract_label_and_features(df_train, hours_ahead)
        x_train = x_train2.values
        y_train = y_train2.values

        # prepare the features list
        print("prepare the features list", end=', ')
        features = list(x_train2.columns)
        mt_arr = list(mutual_info_regression(x_train, y_train.ravel()))
        features_mt = list(zip(features, mt_arr))
        features_mt = sorted(features_mt, key=lambda s: s[1])
        print("select the best features to work with", end=', ')
        best_features_arr = [a for a, b in features_mt]
        best._set_value(str(hours_ahead) + 'AHEAD', 1, best_features_arr)
        best.to_csv('data/best_features.csv', header=False)
    else:
        print("using already calculated best features to work with", end=', ')

if __name__ == '__main__':
    best_num_arr = [5,10,20,50]
    hours_ahead_array = [1,2,5,10,20,24,30]
    save_graphs = True
    # best_num_arr = [5,10]
    # hours_ahead_array = [1,2]
    force_recalculate_best = False
    for best_num in best_num_arr:
        result = pd.DataFrame()
        differences = pd.DataFrame()
        avgs = []
        for hours_ahead in hours_ahead_array:
            mse_csv = pd.read_csv('results/mse_results.csv', index_col='hours ahead')
            mae_csv = pd.read_csv('results/mae_results.csv', index_col='hours ahead')
            create_best(hours_ahead)
            # prepare training data
            print("prepare training data", end=', ')
            df_features = pd.read_csv('data/best_features.csv', index_col=0, header=None)
            # df_features[1] = df_features[1].apply(lambda x: x[1:-1].split(','))
            features = df_features._get_value(str(hours_ahead) + 'AHEAD',1)
            features = features.strip("[\"''\"]").split("', '")
            features = features[(-best_num):]
            # features.append(str(hours_ahead) + 'AHEAD wavepower')
            df_train = pd.read_csv('data/2018_input.csv')

            x_train3 = df_train[features]
            y_train3 = df_train[str(hours_ahead) + 'AHEAD wavepower']

            # x_train3, y_train3 = extract_label_and_features(df_train, hours_ahead)
            x_train = x_train3.values
            y_train = y_train3.values
            # prepare testing data
            print("prepare testing data", end=', ')
            df_train = pd.read_csv('data/2019_input.csv')

            x_test3 = df_train[features]
            y_test3 = df_train[str(hours_ahead) + 'AHEAD wavepower']

            # x_test3, y_test3 = extract_label_and_features(df_test, hours_ahead)
            x_test = x_test3.values
            y_test = y_test3.values

            scaler = StandardScaler()
            scaler.fit(x_train)
            x_train = scaler.transform(x_train)
            x_test = scaler.transform(x_test)

            # put the real results
            print("put the real results.")
            result, differences = add_to_result(result, differences, 'real values', hours_ahead, y_test)

            # calculate ENET
            print("calculate ENET", end="\t# ")
            reg = enet_regressor_build(x_train, y_train)
            predictions = regressor_predict(reg, x_test)
            # put the ENET predictions
            print("put the ENET predictions")
            result, differences = add_to_result(result, differences, 'enet predictions', hours_ahead, predictions)
            make_plot(predictions, y_test, 'enet', best_num, hours_ahead)

            # calculate LIN
            print("calculate LIN", end="\t# ")
            reg = lin_regressor_build(x_train, y_train)
            predictions = regressor_predict(reg, x_test)
            # put the LIN predictions
            print("put the LIN predictions")
            result, differences = add_to_result(result, differences, 'lin predictions', hours_ahead, predictions)
            make_plot(predictions, y_test, 'lin', best_num, hours_ahead)

            # calculate SGD
            print("calculate SGD", end="\t# ")
            reg = sgd_regressor_build(x_train, y_train)
            predictions = regressor_predict(reg, x_test)
            # put the SGD predictions
            print("put the SGD predictions")
            result, differences = add_to_result(result, differences, 'sgd predictions', hours_ahead, predictions)
            make_plot(predictions, y_test, 'sgd', best_num, hours_ahead)

            # calculate DTREE
            print("calculate DTREE", end="\t# ")
            reg = dtree_regressor_build(x_train, y_train)
            predictions = regressor_predict(reg, x_test)
            # put the DTREE predictions
            print("put the DTREE predictions")
            result, differences = add_to_result(result, differences, 'dtree predictions', hours_ahead, predictions)
            make_plot(predictions, y_test, 'dtree', best_num, hours_ahead)

            # calculate KNN
            print("calculate KNN", end="\t# ")
            reg = knn_regressor_build(x_train, y_train)
            predictions = regressor_predict(reg, x_test)
            # put the KNN predictions
            print("put the KNN predictions")
            result, differences = add_to_result(result, differences, 'knn predictions', hours_ahead, predictions)
            make_plot(predictions, y_test, 'knn', best_num, hours_ahead)

            # calculate K-MEANS
            print("calculate K-MEANS", end="\t# ")
            reg, clusters_features, clusters_labels, cluster_num = kmeans_regressor_build(x_train, y_train)
            predictions = kmeans_knn_regressor_predict(reg, clusters_features, clusters_labels, x_test, cluster_num)
            # put the K-MEANS predictions
            print("put the K-MEANS predictions")
            result, differences = add_to_result(result, differences, 'kmeans-knn predictions', hours_ahead, predictions)
            make_plot(predictions, y_test, 'kmeans-knn', best_num, hours_ahead)

            # calculate K-MEANS-SGD
            print("calculate K-MEANS-SGD", end="\t# ")
            reg, clusters_features, clusters_labels, cluster_num = kmeans_regressor_build(x_train, y_train)
            predictions = kmeans_sgd_regressor_predict(reg, clusters_features, clusters_labels, x_test, cluster_num)
            # put the K-MEANS-SGD predictions
            print("put the K-MEANS-SGD predictions")
            result, differences = add_to_result(result, differences, 'kmeans-sgd predictions', hours_ahead, predictions)
            make_plot(predictions, y_test, 'kmeans-sgd', best_num, hours_ahead)

            # calculate K-MEANS-LIN
            print("calculate K-MEANS-LIN", end="\t# ")
            reg, clusters_features, clusters_labels, cluster_num = kmeans_regressor_build(x_train, y_train)
            predictions = kmeans_lin_regressor_predict(reg, clusters_features, clusters_labels, x_test, cluster_num)
            # put the K-MEANS-LIN predictions
            print("put the K-MEANS-LIN predictions")
            result, differences = add_to_result(result, differences, 'kmeans-lin predictions', hours_ahead, predictions)
            make_plot(predictions, y_test, 'kmeans-lin', best_num, hours_ahead)

            print('results for ' + str(hours_ahead) + ' hours ahead with ' + str(best_num) + ' best features:')
            avgs.append(print_results(hours_ahead, 'knn'))
            avgs.append(print_results(hours_ahead, 'kmeans-knn'))
            avgs.append(print_results(hours_ahead, 'kmeans-sgd'))
            avgs.append(print_results(hours_ahead, 'kmeans-lin'))
            avgs.append(print_results(hours_ahead, 'sgd'))
            avgs.append(print_results(hours_ahead, 'dtree'))
            avgs.append(print_results(hours_ahead, 'lin'))
            avgs.append(print_results(hours_ahead, 'enet'))

            print()
            mse_csv.to_csv('results/mse_results.csv', index='hours ahead')
            mae_csv.to_csv('results/mae_results.csv', index='hours ahead')
        types = list(set([t for t,_,_,_ in avgs]))

        for type in types:
            mae_for_type = [mae for t,mae,_,_ in avgs if t == type]
            plt.plot(hours_ahead_array, mae_for_type, label=type)
        plt.legend()
        plt.title('MAE with ' + str(best_num) + ' best features')
        if save_graphs:
            plt.savefig('results/MAE_' + str(best_num) + '_best.png')
        plt.show()
        for type in types:
            mse_for_type = [mse for t,_,mse,_ in avgs if t == type]
            plt.plot(hours_ahead_array, mse_for_type, label=type)
        plt.legend()
        plt.title('MSE with ' + str(best_num) + ' best features')
        if save_graphs:
            plt.savefig('results/MSE_' + str(best_num) + '_best.png')
        plt.show()

        # print all of the results to files
        # result.to_csv('results.csv', index=False)
        # differences.to_csv('differences.csv', index=False)

