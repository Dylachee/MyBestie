import random
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

SEED = 1

random.seed(SEED)
np.random.seed(SEED)

train = pd.read_csv('train_lr.csv')
train = train.sort_values(by=['Word', 'Affix', 'Tag'])
print(train.shape)

test = pd.read_csv('test_lr.csv')
test = test.sort_values(by=['Word', 'Affix', 'Tag'])
print(test.shape)

# Ограничение количества слов до 1000
train = train[:1000]
test = test[:1000]

df = pd.concat([train, test], ignore_index=True)

X = df[['Word', 'Root', 'Affix', 'PoS_root', 'PoS_word']].copy()
y = df['Tag'].copy()

X_pr = pd.get_dummies(X)

le = LabelEncoder()
y = le.fit_transform(y)

train_X = X_pr.loc[train.index, :].copy()
train_y = y[:train.shape[0]].copy()

test_X = X_pr.loc[test.index, :].copy()
test_y = y[train.shape[0]:].copy()

train_X = X_pr.loc[X_pr.index.isin(train.index), :].copy()

lr = LogisticRegression()
lr.fit(train_X, train_y)

lr_predict_result = lr.predict(test_X)
print("F1 score:", f1_score(test_y, lr_predict_result, average='micro'))

SUBMISSION_FILE = 'my_submission.csv'

predictions = lr.predict(test_X)

test['Tag'] = predictions
test[['Word', 'Root', 'Affix', 'Tag']].to_csv(SUBMISSION_FILE, index=False, header=True)

#Не работает 