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

Вдохновлён кодовой базой проекта [AlexeySorokin/NeuralMorphemeSegmentation](https://github.com/AlexeySorokin/NeuralMorphemeSegmentation), который
реализован в рамках
публикации "[Deep Convolutional Networks for Supervised Morpheme Segmentation of Russian Language](https://github.com/AlexeySorokin/NeuralMorphemeSegmentation/blob/master/Articles/MorphemeSegmentation_final.pdf)"
за авторством Алексея Сорокина и Анастасии Кравцовой.

## Примеры

Пример работы модели:

```shell
В	в:ROOT	98.59
воскресенье	воскрес:ROOT/ень:SUFF/е:END	99.30 96.58 100.00
мы	мы:ROOT	99.77
решили	решил:ROOT/и:END	85.80 100.00
перезапланировать	пере:PREF/за:PREF/план:ROOT/ир:SUFF/ова:SUFF/ть:SUFF	100.00 77.91 98.43 100.00 99.98 98.37
```

Или в формате JSONL:

```json lines
{"word": "В", "morphemes": [{"text": "в", "type": "ROOT", "prob": "98.59"}]}
{"word": "воскресенье", "morphemes": [{"text": "воскрес", "type": "ROOT", "prob": "99.3"}, {"text": "ень", "type": "SUFF", "prob": "96.58"}, {"text": "е", "type": "END", "prob": "100.0"}]}
{"word": "мы", "morphemes": [{"text": "мы", "type": "ROOT", "prob": "99.77"}]}
{"word": "решили", "morphemes": [{"text": "решил", "type": "ROOT", "prob": "85.8"}, {"text": "и", "type": "END", "prob": "100.0"}]}
{"word": "перезапланировать", "morphemes": [{"text": "пере", "type": "PREF", "prob": "100.0"}, {"text": "за", "type": "PREF", "prob": "77.91"}, {"text": "план", "type": "ROOT", "prob": "98.43"}, {"text": "ир", "type": "SUFF", "prob": "100.0"}, {"text": "ова", "type": "SUFF", "prob": "99.98"}, {"text": "ть", "type": "SUFF", "prob": "98.37"}]}
```

## Установка и запуск

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

## Как пользоваться

### Тренировка модели

```shell
python3 train.py config/ruMorpheme.json
```

По завершению тренировки будут созданы:

- `model/pytorch-model.bin` - веса модели
- `model/config.json` - конфигурация модели
- `model/vocab.json` - словарь необходимый для работы предикшена

### Валидация модели

```shell
python3 eval.py config/ruMorpheme.json
```

Отчёт валидации будет в `models/evaluation_report.txt`.

### Использование модели

Запуск тестового предикшена из файла [input_text.txt](./input_text.txt):

```shell
python predict.py input_text.txt --model-path=evilfreelancer/ruMorpheme-v0.1
```

Если не указывать `--model-path` то модель и конфигурация будут прочитаны из директории `./model`.
