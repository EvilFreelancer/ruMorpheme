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

Веса модели [evilfreelancer/ruMorpheme-v0.1](https://huggingface.co/evilfreelancer/ruMorpheme-v0.1) на HuggingFace.

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
{"word": "В", "morphemes": [{"text": "в", "type": "ROOT", "prob": "98.59"}]}
{"word": "воскресенье", "morphemes": [{"text": "воскрес", "type": "ROOT", "prob": "99.3"}, {"text": "ень", "type": "SUFF", "prob": "96.58"}, {"text": "е", "type": "END", "prob": "100.0"}]}
{"word": "мы", "morphemes": [{"text": "мы", "type": "ROOT", "prob": "99.77"}]}
{"word": "решили", "morphemes": [{"text": "решил", "type": "ROOT", "prob": "85.8"}, {"text": "и", "type": "END", "prob": "100.0"}]}
{"word": "перезапланировать", "morphemes": [{"text": "пере", "type": "PREF", "prob": "100.0"}, {"text": "за", "type": "PREF", "prob": "77.91"}, {"text": "план", "type": "ROOT", "prob": "98.43"}, {"text": "ир", "type": "SUFF", "prob": "100.0"}, {"text": "ова", "type": "SUFF", "prob": "99.98"}, {"text": "ть", "type": "SUFF", "prob": "98.37"}]}
```

А вот так её можно будет заставить выводить результат:

```shell
В	в:ROOT	98.59
воскресенье	воскрес:ROOT/ень:SUFF/е:END	99.30 96.58 100.00
мы	мы:ROOT	99.77
решили	решил:ROOT/и:END	85.80 100.00
перезапланировать	пере:PREF/за:PREF/план:ROOT/ир:SUFF/ова:SUFF/ть:SUFF	100.00 77.91 98.43 100.00 99.98 98.37
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
