import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix


def main():
    df = pd.read_csv('KNN_Project_Data')
    # print(df.head())
    # sb.pairplot(data=df, hue='TARGET CLASS')
    # plt.show()
    scaler = StandardScaler()
    data = df.drop('TARGET CLASS', axis=1)
    scaler.fit(data)
    scaled_features = scaler.transform(data)
    df_feat = pd.DataFrame(scaled_features, columns=df.columns[:-1])
    # print(df_feat.head())
    X_train, X_test, y_train, y_test = train_test_split(df_feat, df['TARGET CLASS'], test_size=0.3, random_state=101)
    remodel(1, X_train, X_test, y_train, y_test)
    max_k = 40
    error_rate = []
    for i in range(1, max_k):
        model = get_errors(i, X_train, X_test, y_train, y_test)
        error_rate.append(model)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_k), error_rate, color='blue', linestyle='dashed', marker='o', markerfacecolor='red',
             markersize=10)
    plt.title('Error Rate vs K Value')
    plt.xlabel('K')
    plt.ylabel('Error Rate')
    plt.show()
    min_k = np.argmin(error_rate)
    print("Minimum K = "+str(min_k))
    remodel(min_k, X_train, X_test, y_train, y_test)


def get_errors(i, X_Train, X_Test, y_train, y_test):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_Train, y_train)
    prediction = knn.predict(X_Test)
    return np.mean(prediction != y_test)


def remodel(k, X_train, X_test, y_train, y_test):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)
    print("Confusion Matrix")
    print(confusion_matrix(y_test, predictions))
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))


if __name__ == '__main__':
    main()
