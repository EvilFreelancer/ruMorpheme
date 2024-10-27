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
