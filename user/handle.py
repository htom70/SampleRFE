import mariadb
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import RandomUnderSampler
from sklearn.decomposition import PCA,TruncatedSVD
import time


def getAllRecordsFromDatabase(databaseName):
    # connection = mysql.connector.connect(
    #     host="localhost",
    #     user="root",
    #     password="TOmi_1970",
    #     database="retired_transaction")

    connection = mariadb.connect(
        # pool_name="read_pull",
        # pool_size=1,
        host="store.usr.user.hu",
        user="mki",
        password="pwd",
        database=databaseName
    )
    print(connection)
    sql_select_Query = "select * from transaction"
    cursor = connection.cursor()
    cursor.execute(sql_select_Query)
    result = cursor.fetchall()
    # connection.close()
    numpy_array = np.array(result)
    length = len(numpy_array)
    print(f'{databaseName} beolvasva, rekordok száma: {length}')
    return numpy_array[:, :]


if __name__ == '__main__':
    dataset = getAllRecordsFromDatabase("card_100000_1_i")
    dataset_features = dataset[:, 1:-1]
    dataset_labels = dataset[:, -1:]
    sampler=RandomUnderSampler(sampling_strategy=0.5)
    sampledFeatures, sampledLabels = sampler.fit_resample(dataset_features, dataset_labels)
    train_features, test_features, train_labels, test_labels = train_test_split(sampledFeatures, sampledLabels,
                                                                                test_size=0.2, random_state=0)
    # rfe = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=5)
    print("RFECV")
    rfe = RFECV(estimator=DecisionTreeClassifier(), n_jobs=-1)
    rfe.fit(train_features, train_labels)
    n=rfe.n_features_
    print(f'Features: {n}')
    r=rfe.ranking_
    print(f'Ranking: {r}')
    s=rfe.support_
    print(f'Support: {s}')
    reduced_train_features = rfe.transform(train_features)
    reduced_test_features = rfe.transform(test_features)
    rfc = RandomForestClassifier(n_jobs=-1)
    rfc.fit(reduced_train_features, train_labels)
    predicted_labels = rfc.predict(reduced_test_features)
    confusionMatrix = confusion_matrix(test_labels, predicted_labels)
    print(f"Confusion Matrix: {confusionMatrix}")
    print(rfc.feature_importances_)

    print("PCA")
    svd = PCA(n_components=2)
    startOfPCA = time.time()
    svd.fit(train_features)
    pcaTransformed_train_features = svd.transform(train_features)
    pcaTransformed_test_features = svd.transform(test_features)
    endOfPCA = time.time()
    rfc = RandomForestClassifier(n_jobs=-1)
    rfc.fit(pcaTransformed_train_features, train_labels)
    predicted_labels = rfc.predict(pcaTransformed_test_features)
    confusionMatrix = confusion_matrix(test_labels, predicted_labels)
    print(f"Confusion Matrix: {confusionMatrix}")
    print(rfc.feature_importances_)

    print("SVD")
    svd = TruncatedSVD(n_components=2)
    startOfPCA = time.time()
    svd.fit(train_features)
    pcaTransformed_train_features = svd.transform(train_features)
    pcaTransformed_test_features = svd.transform(test_features)
    endOfPCA = time.time()
    rfc = RandomForestClassifier(n_jobs=-1)
    rfc.fit(pcaTransformed_train_features, train_labels)
    predicted_labels = rfc.predict(pcaTransformed_test_features)
    confusionMatrix = confusion_matrix(test_labels, predicted_labels)
    print(f"Confusion Matrix: {confusionMatrix}")
    print(rfc.feature_importances_)


