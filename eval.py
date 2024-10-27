import argparse

from rumorpheme.model import RuMorphemeModel
from rumorpheme.utils import labels_to_morphemes

parser = argparse.ArgumentParser(description="Morpheme Segmentation Script")
parser.add_argument("--model-path", default='./model', help="Local directory containing model files or Hugging Face repo ID")
parser.add_argument("--test-path", default='data/test_Tikhonov_reformat.txt', help="Path to the input file with test cases")
parser.add_argument("--outfile", default='model/evaluation_report.txt', help="Path to save evaluation report")
parser.add_argument("--use-morpheme-types", action='store_false')
args = parser.parse_args()

# Settings
model_path = args.model_path
test_path = args.test_path
outfile = args.outfile
use_morpheme_types = bool(args.use_morpheme_types)

# Load the model
model = RuMorphemeModel.from_pretrained(model_path)
model.eval()

# Read input text
words = []
with open(test_path, "r", encoding="utf8") as f:
    for line in f:
        words.append(line.split('\t')[0])
    words = [ln.strip() for ln in words]

# Preprocess input text
max_word_length = max(len(word) for word in words) + 2  # +2 for BEGIN and END
inputs = words  # Assuming each line contains one word

# Perform predictions
all_predictions, all_log_probs = model.predict(words)

with open(outfile, "w", encoding="utf8") as fout:
    # Process and display the results
    for idx, word in enumerate(words):
        # Convert predictions to morphemes
        morphemes, morphemes_types, morphemes_probs = labels_to_morphemes(
            word.lower(),
            all_predictions[idx],
            all_log_probs[idx],
            use_morpheme_types
        )

        # Combine morphemes and their types with a slash
        morpheme_with_types = [
            f"{morpheme}:{morpheme_type}"
            for morpheme, morpheme_type in zip(morphemes, morphemes_types)
        ]
        # Join morpheme/type pairs with tabs
        morpheme_str = '/'.join(morpheme_with_types)
        probs_str = " ".join(f"{prob:.2f}" for prob in morphemes_probs)
        output_line = f"{word}\t{morpheme_str}\t{probs_str}\n"
        fout.write(output_line)

print(f"Результаты сохранены в {outfile}")
