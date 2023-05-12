import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import f1_score

# Загрузка данных
train_data = pd.read_csv('train_lr.csv')
test_data = pd.read_csv('test_lr.csv')

# Преобразование меток в числовой формат
label_encoder = LabelEncoder()
label_encoder.fit(train_data['Root'])

# Применение преобразования меток к обоим наборам данных
train_labels = label_encoder.transform(train_data['Root'])
test_labels = label_encoder.transform(test_data['Root'])
num_classes = len(label_encoder.classes_)

# Преобразование слов в числовые последовательности
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(train_data['Word'])
train_sequences = tokenizer.texts_to_sequences(train_data['Word'])
test_sequences = tokenizer.texts_to_sequences(test_data['Word'])

# Подготовка входных данных
max_length = max(len(seq) for seq in train_sequences)
train_inputs = pad_sequences(train_sequences, maxlen=max_length, padding='post')
test_inputs = pad_sequences(test_sequences, maxlen=max_length, padding='post')

# Создание модели LSTM
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.index_word)+1, output_dim=64, input_length=max_length))
model.add(LSTM(64))
model.add(Dense(num_classes, activation='softmax'))

# Компиляция и обучение модели
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_inputs, to_categorical(train_labels, num_classes), epochs=10, batch_size=32)

# Предсказание меток для тестовых данных
predictions = model.predict(test_inputs)
predicted_labels = np.argmax(predictions, axis=1)

# Обратное преобразование числовых меток в исходный формат
predicted_labels = label_encoder.inverse_transform(predicted_labels)

# Расчет метрики F1
f1 = f1_score(test_labels, predicted_labels, average='weighted')
print("F1-мера на тестовых данных:", f1)

print(X_pr.index)
print(X_pr.index.isin(train.index))
