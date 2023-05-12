import nltk
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

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
        # Здесь вы можете добавить свои правила разделения на морфемы
    ]

    # Проверка слова на соответствие каждому правилу
    for rule in rules:
        prefixes, suffix_root, suffix_affix = rule
        if word.startswith(tuple(prefixes)):
            root = word[len(prefixes):]
            if root.endswith(suffix_root):
                affix = root[:len(root) - len(suffix_root)]
                return [root, affix]
    return [word]  # Если нет совпадений, возвращаем слово как одну морфему

# Загрузка тренировочных данных
train_data = load_data('train_lr.csv')
train_data = train_data[:1000]  # Ограничение до 1000 строк

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
train_df = pd.DataFrame({'Word': train_tokens, 'Morphemes': train_morphemes})
train_df['Root'] = train_df['Morphemes'].apply(lambda x: x[0])
train_df['Affix'] = train_df['Morphemes'].apply(lambda x: x[1] if len(x) > 1 else '')
train_df['Tag'] = train_labels[:1000]  # Используйте только первые 1000 меток

# Создание датафрейма с тестовыми данными
test_df = pd.DataFrame({'Word': test_tokens, 'Morphemes': test_morphemes})
test_df['Root'] = test_df['Morphemes'].apply(lambda x: x[0])
test_df['Affix'] = test_df['Morphemes'].apply(lambda x: x[1] if len(x) > 1 else '')

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

