import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline

# Importing data
data = pd.read_csv('cancer.csv')
df = pd.DataFrame(data)
df.head()

# Dropping unwanted column
df = df.drop(['id', 'Unnamed: 32'], axis=1)

# No of rows and columns
print(df.shape)

# Except 'diagnosis' all the columns are numeric
df.describe()

# checking for null value
df.isnull().values.any()

# Count class labels
df['diagnosis'].value_counts()

# Data visualization to create histogram
df.hist(bins=50, figsize=(15, 15))
plt.show()

# Finding correlation
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(df.corr(), ax=ax)


# Removing outliers
def remove_outliers_zscore(df, columns, threshold=3):
    for column in columns:
        mean = np.mean(df[column])
        std_dev = np.std(df[column])
        zscores = np.abs((df[column] - mean) / std_dev)
        outliers = df[zscores > threshold]
        df = df.drop(outliers.index)
    return df


df1 = remove_outliers_zscore(df, df.iloc[:,1:11])
df1.shape

# Normalizing the data
standardScaler = StandardScaler()
scale = ['radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean','compactness_mean',
         'concavity_mean', 'concave points_mean','symmetry_mean','fractal_dimension_mean']
df1[scale] = standardScaler.fit_transform(df1[scale])

# Separate labels and features
X = df.drop(columns=['diagnosis'])
y = df['diagnosis']

# Convert the M to 1 and B to 0
label = LabelEncoder()
y = label.fit_transform(y)

# Spilt the train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# we used 30% test data
# check the size before beginning
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# finding accuracy of the SVM classifiers
svm = SVC()
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Precision:', precision_score(y_test, y_pred, pos_label=1))
print('Recall:', recall_score(y_test, y_pred, pos_label=1))
print('F1 score:', f1_score(y_test, y_pred, pos_label=1))

# Hyperparameter tuning
svm_pipe = make_pipeline(StandardScaler(), SVC())
param_grid = {
    'svc__C': [0.1, 1, 10],
    'svc__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'svc__degree': [2, 3, 4],
}
grid_search = GridSearchCV(svm_pipe, param_grid=param_grid, cv=5)
grid_search.fit(X, y)
print("Best parameters: ", grid_search.best_params_)
print("Best score: ", grid_search.best_score_)


