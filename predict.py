import sys
import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from huggingface_hub import hf_hub_download
import argparse

from train import Partitioner, prepare_data, labels_to_morphemes

import re

parser = argparse.ArgumentParser(description="Morpheme Segmentation Script")
parser.add_argument("input_text_file", help="Path to the input text file")
parser.add_argument("--model-path", default='./model', help="Local directory containing model files or Hugging Face repo ID")
parser.add_argument("--batch-size", default=10)
parser.add_argument("--use-morpheme-types", action='store_false')
args = parser.parse_args()

# Settings
model_path = args.model_path
input_text_file = args.input_text_file
batch_size = int(args.batch_size)
use_morpheme_types = bool(args.use_morpheme_types)

if os.path.isdir(model_path):
    # Load files from local directory
    config_file = os.path.join(model_path, "config.json")
    vocab_file = os.path.join(model_path, "vocab.json")
    model_file = os.path.join(model_path, "pytorch-model.bin")
    if not all(os.path.isfile(f) for f in [config_file, vocab_file, model_file]):
        print("Model files not found in the specified directory.")
        sys.exit(1)
else:
    # Assume model_path is a Hugging Face repo ID
    repo_id = model_path
    config_file = hf_hub_download(repo_id=repo_id, filename="config.json")
    vocab_file = hf_hub_download(repo_id=repo_id, filename="vocab.json")
    model_file = hf_hub_download(repo_id=repo_id, filename="pytorch-model.bin")

# Read parameters of model
with open(config_file, "r", encoding="utf8") as fin:
    model_params = json.load(fin)

# Load vocabularies
with open(vocab_file, "rb") as f:
    vocab_data = json.load(f)
symbols = vocab_data["symbols"]
symbol_codes = vocab_data["symbol_codes"]
target_symbols = vocab_data["target_symbols"]
target_symbol_codes = vocab_data["target_symbol_codes"]

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Partitioner(
    symbols_number=len(symbols),
    target_symbols_number=len(target_symbols),
    params=model_params
)
model.load_state_dict(torch.load(model_file, map_location=device, weights_only=True))
model.to(device)
model.eval()
# print(f"Model loaded from {model_path}")

# Read input text
words = []
# Регулярное выражение для проверки русской буквы
russian_letter_pattern = re.compile(r'[а-яА-ЯёЁ]')
with open(input_text_file, "r", encoding="utf8") as f:
    for line in f:
        if line.strip():
            words += line.split(' ')
    words = [ln.strip() for ln in words]
    
    # Обрабатываем каждый элемент в списке words
    for i in range(len(words)):
        word = words[i]
        
        # Удаляем начальные символы, не являющиеся русскими буквами
        while word and not russian_letter_pattern.match(word[0]):
            word = word[1:]
        
        # Удаляем конечные символы, не являющиеся русскими буквами
        while word and not russian_letter_pattern.match(word[-1]):
            word = word[:-1]
        
        # Обновляем элемент списка, если слово не пустое, иначе удаляем его
        words[i] = word if word else None
    
    # Удаляем пустые элементы (None) из списка
    words = [word for word in words if word]

# Preprocess input text
max_word_length = max(len(word) for word in words) + 2  # +2 for BEGIN and END
inputs = words  # Assuming each line contains one word


# Prepare the dataset
class InferenceDataset(Dataset):
    def __init__(self, data, symbol_codes, bucket_length):
        self.inputs = prepare_data(data, symbol_codes, bucket_length)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_seq = self.inputs[idx]
        return torch.tensor(input_seq, dtype=torch.long)


dataset = InferenceDataset(inputs, symbol_codes, max_word_length)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)


# Function to perform predictions
def predict(model, dataloader, device):
    model.eval()
    all_predictions = []
    all_log_probs = []
    with torch.no_grad():
        for inputs in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            log_probs = torch.log_softmax(outputs, dim=-1)
            predictions = torch.argmax(log_probs, dim=-1)
            all_predictions.extend(predictions.cpu().numpy())
            all_log_probs.extend(log_probs.cpu().numpy())
    return all_predictions, all_log_probs


# Perform predictions
all_predictions, all_log_probs = predict(model, dataloader, device)

# Process and display the results
for idx, word in enumerate(words):
    pred_seq = all_predictions[idx]
    log_prob_seq = all_log_probs[idx]

    # Skip BEGIN and END tokens
    morphemes, morpheme_types, morpheme_probs = labels_to_morphemes(
        word.lower(),
        pred_seq[1:-1],
        log_prob_seq[1:-1],
        target_symbols,
        use_morpheme_types
    )

    # Combine morphemes, their types and probs to objects
    results = []
    for morpheme, morpheme_type, morpheme_prob in zip(morphemes, morpheme_types, morpheme_probs):
        results.append({"text": morpheme, "type": morpheme_type, "prob": str(morpheme_prob.round(2))})

    # Return results
    output = {"word": word, "morphemes": results}
    print(json.dumps(output, ensure_ascii=False))
