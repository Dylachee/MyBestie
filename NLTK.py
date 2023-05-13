import nltk
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# Загрузка данных из CSV файла
def load_data(filename):
    data = pd.read_csv(filename)
    return data

# Разбиение слова на морфемы с помощью правил
def split_morphemes(word):
    # Набор правил для разделения слова на морфемы
    rules = [
        # Правило 1: Определение корней и аффиксов на основе заданных префиксов и суффиксов
        (['кыр', 'бол'], 'кыр', 'бол'),  # Пример: кыр + бол -> кыр - корень, бол - аффикс
        (['тур', 'ул'], 'тур', 'ул'),  # Пример: тур + ул -> тур - корень, ул - аффикс
        # Правило 2: Определение корней и аффиксов на основе заданных правил
        (['мо', 'ст'], 'мо', 'ст'),  # Пример: мо + ст -> мо - корень, ст - аффикс
        (['быр', 'ул'], 'быр', 'ул'),  # Пример: быр + ул -> быр - корень, ул - аффикс
        # Правило 3:
        (['пред', 'ост'], 'пред', 'ост'),  # Пример: пред + ост -> пред - корень, ост - аффикс
        (['пере', 'ход'], 'пере', 'ход'),  # Пример: пере + ход -> пере - корень, ход - аффикс
        # Добавьте свои правила ниже
        (['при', 'ход'], 'при', 'ход'),  # Пример: при + ход -> при - корень, ход - аффикс
        (['раз', 'дел'], 'раз', 'дел'),  # Пример: раз + дел -> раз - корень, дел - аффикс
        (['с', 'каз'], 'с', 'каз'),  # Пример: с + каз -> с - корень, каз - аффикс
        (['по', 'ход'], 'по', 'ход'),  # Пример: по + ход -> по - корень, ход - аффикс
    ]

    # Проверка слова на соответствие каждому правилу
    for rule in rules:
        prefixes, suffix_root, suffix_affix = rule
        if word.startswith(tuple(prefixes)):
            root = word[len(prefixes):]
            if root.endswith(suffix_root):
                affix = root[:len(root) - len(suffix_root)]
                return [root, affix]
    return [word]


# Загрузка тренировочных данных
train_data = load_data('train_lr.csv')
train_data = train_data[:10]  # Ограничение до 1000 строк

# Загрузка тестовых данных
test_data = load_data('test_lr.csv')

# Извлечение слов и их меток из данных
train_words = train_data['Word'].tolist()[:1000]  # Используйте только первые 1000 слов
train_labels = train_data['Tag'].tolist()[:1000]  # Используйте только первые 1000 меток
test_words = test_data['Word'].tolist()  # Получите все слова из тестовых данных

# Токенизация слов
train_tokens = nltk.word_tokenize(' '.join(train_words))
test_tokens = nltk.word_tokenize(' '.join(test_words))

# Разбиение слов на морфемы
train_morphemes = [split_morphemes(word) for word in train_tokens]
test_morphemes = [split_morphemes(word) for word in test_tokens]

# Создание датафрейма с тренировочными данными
train_df = pd.DataFrame({'Word': train_tokens[:1000], 'Morphemes': train_morphemes[:1000]})
train_df['Root'] = train_df['Morphemes'].apply(lambda x: x[0])
train_df['Affix'] = train_df['Morphemes'].apply(lambda x: x[1] if len(x) > 1 else '')

# Создание датафрейма с тестовыми данными
test_df = pd.DataFrame({'Word': test_tokens, 'Morphemes': test_morphemes})
test_df['Root'] = test_df['Morphemes'].apply(lambda x: x[0])
test_df['Affix'] = test_df['Morphemes'].apply(lambda x: x[1] if len(x) > 1 else '')

# Создание экземпляра OneHotEncoder
one_hot_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')

# Объединение тренировочного и тестового датасетов для одновременной обработки признаков
combined_df = pd.concat([train_df, test_df])

# One-Hot Encoding для столбцов 'Root' и 'Affix'
one_hot_encoded = one_hot_encoder.fit_transform(combined_df[['Root', 'Affix']])

# Разделение обратно на тренировочный и тестовый датасеты
train_one_hot_encoded = one_hot_encoded[:len(train_df)]
test_one_hot_encoded = one_hot_encoded[len(train_df):]

# Замена столбцов 'Root' и 'Affix' в тренировочном и тестовом датасетах
train_df.drop(['Root', 'Affix'], axis=1, inplace=True)
train_df = pd.concat([train_df, pd.DataFrame(train_one_hot_encoded)], axis=1)

test_df.drop(['Root', 'Affix'], axis=1, inplace=True)
test_df = pd.concat([test_df, pd.DataFrame(test_one_hot_encoded)], axis=1)

# Обработка и преобразование данных
train_X = train_df[['Root', 'Affix']]
train_y = train_df['Tag']
test_X = test_df[['Root', 'Affix']]

# Обучение модели
lr = LogisticRegression()
lr.fit(train_X, train_y)

# Предсказываем
predictions = lr.predict(test_X)

# Записываем результаты в CSV
submission_file = 'my_submission.csv'
test_df['Tag'] = predictions
test_df[['Word', 'Root', 'Affix', 'Tag']].to_csv(submission_file, index=False, header=True)
