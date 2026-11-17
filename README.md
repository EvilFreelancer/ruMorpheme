# ruMorpheme - Russian Morphemes Segmentation

Проект языковой модели для проведения морфемного анализа и сегментации слов русского языка.

Обученная модель способна сегментировать слова, выделяя в них:

- приставки (PREF)
- корни (ROOT)
- соединительные гласные (LINK)
- дефисы (HYPH)
- суффиксы (SUFF)
- постфиксы (POSTFIX)
- окончания (END)

По умолчанию библиотека использует веса [evilfreelancer/ruMorpheme-v0.1](https://huggingface.co/evilfreelancer/ruMorpheme-v0.1). Версия [v0.2](https://huggingface.co/evilfreelancer/ruMorpheme-v0.2) доступна отдельно.

Вдохновлён кодовой базой
проекта [AlexeySorokin/NeuralMorphemeSegmentation](https://github.com/AlexeySorokin/NeuralMorphemeSegmentation), который
реализован в рамках
публикации "[Deep Convolutional Networks for Supervised Morpheme Segmentation of Russian Language](https://github.com/AlexeySorokin/NeuralMorphemeSegmentation/blob/master/Articles/MorphemeSegmentation_final.pdf)"
за авторством Алексея Сорокина и Анастасии Кравцовой.

## Установка

Проект доступен через PyPi, и его можно установить с помощью pip:

```shell
pip install rumorpheme
```

## Примеры использования

После установки можно использовать модель для сегментации морфем с помощью следующего скрипта:

```python
import sys
import json
from rumorpheme.model import RuMorphemeModel
from rumorpheme.utils import labels_to_morphemes

# Чтение входных слов из аргументов командной строки
words = sys.argv[1:]  # Список слов, переданных через командную строку

# Загрузка модели
model = RuMorphemeModel.from_pretrained("evilfreelancer/ruMorpheme-v0.1")
model.to("cuda")
model.eval()

# Инференс
all_predictions, all_log_probs = model.predict(words)

# Обработка и отображение результатов
for idx, word in enumerate(words):
    morphs, morph_types, morph_probs = labels_to_morphemes(
        word.lower(),
        all_predictions[idx],
        all_log_probs[idx]
    )

    results = []
    for morpheme, morpheme_type, morpheme_prob in zip(morphs, morph_types, morph_probs):
        results.append({"text": morpheme, "type": morpheme_type, "prob": str(morpheme_prob.round(2))})

    output = {"word": word, "morphemes": results}
    print(json.dumps(output, ensure_ascii=False))
```

## Пример работы модели

В случае если вы используете скрипт предикшена из примера выше, то результат будет выглядеть следующим образом:

```json lines
{"word": "В", "morphemes": [{"text": "в", "type": "UNKNOWN", "prob": "100.0"}]}
{"word": "воскресенье", "morphemes": [{"text": "воскресе", "type": "ROOT", "prob": "99.94"}, {"text": "нье", "type": "SUFF", "prob": "93.6"}]}
{"word": "мы", "morphemes": [{"text": "мы", "type": "UNKNOWN", "prob": "100.0"}]}
{"word": "решили", "morphemes": [{"text": "реши", "type": "ROOT", "prob": "97.5"}, {"text": "л", "type": "SUFF", "prob": "88.58"}, {"text": "и", "type": "ROOT", "prob": "75.63"}]}
{"word": "перезапланировать", "morphemes": [{"text": "перез", "type": "PREF", "prob": "100.0"}, {"text": "ап", "type": "PREF", "prob": "88.38"}, {"text": "лани", "type": "ROOT", "prob": "99.69"}, {"text": "ро", "type": "SUFF", "prob": "100.0"}, {"text": "ват", "type": "SUFF", "prob": "99.91"}, {"text": "ь", "type": "SUFF", "prob": "99.98"}]}
```

А вот так её можно будет заставить выводить результат:

```shell
В	в:UNKNOWN	100.00
воскресенье	воскресе:ROOT/нье:SUFF	99.94 93.60
мы	мы:UNKNOWN	100.00
решили	реши:ROOT/л:SUFF/и:ROOT	97.50 88.58 75.63
перезапланировать	перез:PREF/ап:PREF/лани:ROOT/ро:SUFF/ват:SUFF/ь:SUFF	100.00 88.38 99.69 100.00 99.91 99.98
```

Если форматировать вывод:

```python
# Обработка и отображение результатов
for idx, word in enumerate(words):
    morphs, morph_types, morph_probs = labels_to_morphemes(
        word.lower(),
        all_predictions[idx],
        all_log_probs[idx]
    )

    # Комбинируем морфемы и их типы через косую черту
    morpheme_with_types = [
        f"{morpheme}:{morpheme_type}"
        for morpheme, morpheme_type in zip(morphs, morph_types)
    ]

    # Добавляем вероятности к морфемам
    morpheme_str = '/'.join(morpheme_with_types)
    probs_str = " ".join(f"{prob:.2f}" for prob in morph_probs)
    output_line = f"{word}\t{morpheme_str}\t{probs_str}\n"
    print(output_line)
```

## Про ручное обучение

Склонируем проект и подготовим окружение:

```shell
git clone https://github.com/EvilFreelancer/ruMorpheme.git
cd ruMorpheme
python3 -m venv venv
pip install -r requirements.txt
```

Активируем окружение:

```shell
source venv/bin/activate
```

### Тренировка модели

```shell
python3 train.py config/ruMorpheme.json
```

По завершению тренировки будут созданы:

- `model/pytorch_model.bin` - веса модели
- `model/config.json` - конфигурация модели
- `model/vocab.json` - словарь необходимый для работы предикшена

### Валидация обученной модели

```shell
python3 eval.py
```

Отчёт валидации будет в `models/evaluation_report.txt`.

### Инференс обученной модели

Запуск тестового инференса из файла [input_text.txt](./input_text.txt):

```shell
python predict_file.py input_text.txt --model-path=evilfreelancer/ruMorpheme-v0.1
```

Если не указывать `--model-path` то модель и конфигурация будут прочитаны из директории `./model`.

## Лицензия

Этот проект лицензирован под лицензией `MIT`. Подробности см. в файле [LICENSE](./LICENSE).

## Цитирование

```
@misc{rumorpheme2024sources,
    title={ruMorpheme - Russian Morphemes Segmentation},
    author={Pavel Rykov},
    year={2024}
}
```
