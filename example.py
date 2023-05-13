import random
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

# Фиксируем генераторы случайных чисел для воспроизводимости
SEED = 1
random.seed(SEED)
np.random.seed(SEED)

# Загрузка данных
train = pd.read_csv('train_lr.csv')
train = train.sort_values(by=['Word', 'Affix', 'Tag'])
print(train.shape)

test = pd.read_csv('test_lr.csv')
test = test.sort_values(by=['Word', 'Affix', 'Tag'])
print(test.shape)

# Объединение тренировочных и тестовых данных
df = pd.concat([train, test], ignore_index=True)

# Извлечение признаков и целевой переменной
X = df[['Word', 'Root', 'Affix', 'PoS_root', 'PoS_word']].copy()
y = df['Tag'].copy()

# Преобразование признаков с помощью One-Hot Encoding
X_pr = pd.get_dummies(X)

# Кодирование целевой переменной
le = LabelEncoder()
y = le.fit_transform(y)

# Разделение на тренировочный и тестовый наборы данных
train_X = X_pr.loc[train.index, :].copy()
train_y = y[:train.shape[0]].copy()

test_X = X_pr.loc[test.index, :].copy()
test_y = y[train.shape[0]:].copy()

train_X.shape, train_y.shape

# Обучение модели
lr = LogisticRegression()
lr.fit(train_X, train_y)

# Прогнозирование на тестовом наборе данных
lr_predict_result = lr.predict(test_X)

print("F1 score: ", f1_score(test_y, lr_predict_result, average='micro'))

# Предсказание
SUBMISSION_FILE = 'my_submission.csv'
predictions = lr.predict(test_X)

# Запись результатов в CSV
test['Tag'] = predictions
test[['Word', 'Root', 'Affix', 'Tag']].to_csv(SUBMISSION_FILE, index=False, header=True)
