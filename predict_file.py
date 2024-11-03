import json
import argparse
import re

from rumorpheme.model import RuMorphemeModel
from rumorpheme.utils import labels_to_morphemes

parser = argparse.ArgumentParser(description="Morpheme Segmentation Script")
parser.add_argument("input_text_file", help="Path to the input text file")
parser.add_argument("--model-path", default='./model', help="Local directory containing model files or Hugging Face repo ID")
parser.add_argument("--use-morpheme-types", action='store_false')
args = parser.parse_args()

# Settings
model_path = args.model_path
input_text_file = args.input_text_file
use_morpheme_types = bool(args.use_morpheme_types)

# Load the model
print(f"Loading model from {model_path}...")
model = RuMorphemeModel.from_pretrained(model_path)
model.eval()

# Read input text
words = []
with open(input_text_file, "r", encoding="utf8") as f:
    for line in f:
        if line.strip():
            words += line.split(' ')
    words = [re.sub(r'[^а-яА-Я\-]', '', word).lower() for word in words]

# Perform predictions
all_predictions, all_log_probs = model.predict(words)

# Process and display the results
for idx, word in enumerate(words):
    # Convert predictions to morphemes
    morphemes, morphemes_types, morphemes_probs = labels_to_morphemes(
        word.lower(),
        all_predictions[idx],
        all_log_probs[idx],
        use_morpheme_types
    )

    # Combine morphemes, their types and probs to objects
    results = []
    for morpheme, morpheme_type, morpheme_prob in zip(morphemes, morphemes_types, morphemes_probs):
        results.append({"text": morpheme, "type": morpheme_type, "prob": str(morpheme_prob.round(2))})

    # Return results
    output = {"word": word, "morphemes": results}
    print(json.dumps(output, ensure_ascii=False))
