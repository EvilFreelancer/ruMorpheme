import sys
import json
import torch
from torch.utils.data import Dataset, DataLoader

from train import Partitioner, prepare_data, labels_to_morphemes

# Load the configuration file
if len(sys.argv) < 3:
    print("Usage: python predict.py <config_file.json> <input_text_file>")
    sys.exit(1)

config_file = sys.argv[1]
input_text_file = sys.argv[2]

with open(config_file, "r", encoding="utf8") as fin:
    params = json.load(fin)

# Settings
vocab_path = params.get("vocab_file", "model/vocab.json")
model_params = params.get("model_params")
model_file = params.get("model_file", "model/pytorch-model.bin")
batch_size = params.get("batch_size", 32)
use_morpheme_types = params.get("use_morpheme_types", True)

# Load vocabularies
with open(vocab_path, "rb") as f:
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
model.load_state_dict(torch.load(model_file, map_location=device))
model.to(device)
model.eval()
# print(f"Model loaded from {model_path}")

# Read input text
words = []
with open(input_text_file, "r", encoding="utf8") as f:
    for line in f:
        if line.strip():
            words += line.split(' ')
    words = [ln.strip() for ln in words]

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
