import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from custom_logistic_regression import sigmoid, gradient_descent, gradient_logistic, predict

cancer = load_breast_cancer()

x = cancer.data#type:ignore
y = cancer.target#type:ignore

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.3)

w = np.zeros((x.shape[1], ))
b = 0

w, b = gradient_descent(x_train, y_train, w, b, 0.01, 1000)

y_pred_custom = (predict(x_test, w, b) >= 0.5).astype(int)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)
y_pred_sklearn = model.predict(x_test)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_custom))
print(classification_report(y_test, y_pred_sklearn))

import pandas as pd
cancer_df = pd.DataFrame(cancer.data, columns = cancer.feature_names)#type: ignore
cancer_df['target'] = cancer.target#type: ignore

# Correlation
corr = cancer_df.corr()['target']
corr = corr[0:-1]
plt.barh(cancer_df.drop(columns=['target']).columns, width=corr)
plt.show()

# Heatmap of Correlation
from seaborn import heatmap
corr = cancer_df.corr()
heatmap(cancer_df.corr(), cmap = 'viridis')