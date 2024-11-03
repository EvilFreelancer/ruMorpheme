import os
import sys
import bisect
import numpy as np
import json
from itertools import chain
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import wandb
from read import read_BMES, read_splitted

from rumorpheme.model import RuMorphemeModel
from rumorpheme.utils import prepare_data
from rumorpheme.const import PAD, AUXILIARY


def read_config(infile):
    with open(infile, "r", encoding="utf8") as fin:
        config = json.load(fin)
    if "use_morpheme_types" not in config:
        config["use_morpheme_types"] = True
    return config


def to_one_hot(data, classes_number):
    return np.eye(classes_number, dtype=np.uint8)[data]


def _make_vocabulary(source):
    symbols = {a for word in source for a in word}
    symbols.add('-')  # Добавляем дефис в множество символов
    symbols = AUXILIARY + sorted(symbols)
    symbol_codes = {a: i for i, a in enumerate(symbols)}
    return symbols, symbol_codes


def collect_buckets(lengths, buckets_number, max_bucket_size=-1):
    m = len(lengths)
    lengths = sorted(lengths)
    bucket_lengths = []
    last_bucket_length = 0
    for i in range(buckets_number):
        level = (m * (i + 1) // buckets_number) - 1
        curr_length = lengths[level]
        if curr_length > last_bucket_length:
            bucket_lengths.append(curr_length)
            last_bucket_length = curr_length
    indexes = [[] for _ in bucket_lengths]
    for i, length in enumerate(lengths):
        index = bisect.bisect_left(bucket_lengths, length)
        indexes[index].append(i)
    if max_bucket_size != -1:
        bucket_lengths = list(chain.from_iterable(
            ([L] * ((len(curr_indexes) - 1) // max_bucket_size + 1))
            for L, curr_indexes in zip(bucket_lengths, indexes)
            if len(curr_indexes) > 0))
        indexes = [curr_indexes[start:start + max_bucket_size]
                   for curr_indexes in indexes
                   for start in range(0, len(curr_indexes), max_bucket_size)]
    return [(L, curr_indexes) for L, curr_indexes in zip(bucket_lengths, indexes) if len(curr_indexes) > 0]


class MorphDataset(Dataset):
    def __init__(self, data, targets, symbol_codes, target_symbol_codes, bucket_length):
        self.inputs = prepare_data(data, symbol_codes, bucket_length)
        self.targets = prepare_data(targets, target_symbol_codes, bucket_length)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_seq = self.inputs[idx]
        target_seq = self.targets[idx]
        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(target_seq, dtype=torch.long)


def train_model(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        outputs = outputs.view(-1, outputs.shape[-1])
        targets = targets.view(-1)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            outputs = outputs.view(-1, outputs.shape[-1])
            targets = targets.view(-1)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    return total_loss / len(dataloader)


def predict(model, dataloader, device):
    model.eval()
    all_predictions = []
    all_log_probs = []
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            log_probs = torch.log_softmax(outputs, dim=-1)
            predictions = torch.argmax(log_probs, dim=-1)
            all_predictions.extend(predictions.cpu().numpy())
            all_log_probs.extend(log_probs.cpu().numpy())
    return all_predictions, all_log_probs


def measure_quality(targets, predicted_targets, measure_last=True):
    """
    targets: метки корректных ответов
    predicted_targets: метки предсказанных ответов

    Возвращает словарь со значениями основных метрик
    """

    TP, FP, FN, equal, total = 0, 0, 0, 0, 0
    SE = ['{}-{}'.format(x, y) for x in "SE" for y in ["ROOT", "PREF", "SUFF", "END", "LINK", "None"]]
    corr_words = 0

    for corr, pred in zip(targets, predicted_targets):
        corr_len = len(corr) + int(measure_last) - 1
        pred_len = len(pred) + int(measure_last) - 1
        boundaries = [i for i in range(corr_len) if corr[i] in SE]
        pred_boundaries = [i for i in range(pred_len) if pred[i] in SE]
        common = [x for x in boundaries if x in pred_boundaries]
        TP += len(common)
        FN += len(boundaries) - len(common)
        FP += len(pred_boundaries) - len(common)
        equal += sum(int(x == y) for x, y in zip(corr, pred))
        total += len(corr)
        corr_words += (corr == pred).all()

    precision = TP / (TP + FP) if TP + FP > 0 else 0.0
    recall = TP / (TP + FN) if TP + FN > 0 else 0.0
    f1 = TP / (TP + 0.5 * (FP + FN)) if TP + FP + FN > 0 else 0.0
    accuracy = equal / total if total > 0 else 0.0
    word_accuracy = corr_words / len(targets) if len(targets) > 0 else 0.0

    return {
        "Точность": precision,
        "Полнота": recall,
        "F1-мера": f1,
        "Корректность": accuracy,
        "Точность по словам": word_accuracy
    }


if __name__ == "__main__":
    np.random.seed(261)
    if len(sys.argv) < 2:
        sys.exit("Укажите файл конфигурации")
    config_file = sys.argv[1]
    params = read_config(config_file)
    use_morpheme_types = params.get("use_morpheme_types", True)
    measure_last = params.get("measure_last", use_morpheme_types)
    read_func = read_BMES if use_morpheme_types else read_splitted

    # Load data
    if "train_file" in params:
        n = params.get("n_train")
        inputs, targets = read_func(params["train_file"], n=n)
        if "dev_file" in params:
            n = params.get("n_dev")
            dev_inputs, dev_targets = read_func(params["dev_file"], n=n)
        else:
            dev_inputs, dev_targets = None, None
    else:
        inputs, targets, dev_inputs, dev_targets = None, None, None, None

    # Build vocabularies
    symbols, symbol_codes = _make_vocabulary(inputs)
    target_symbols, target_symbol_codes = _make_vocabulary(targets)

    # Prepare data
    lengths = [len(word) + 2 for word in inputs]  # +2 for BEGIN and END
    buckets_with_indexes = collect_buckets(lengths, buckets_number=10)
    train_data_loaders = []
    for _, bucket_indexes in buckets_with_indexes:
        bucket_inputs = [inputs[i] for i in bucket_indexes]
        bucket_targets = [targets[i] for i in bucket_indexes]
        # Recompute the actual bucket length for this bucket
        actual_bucket_length = max(len(word) for word in bucket_inputs) + 2  # +2 for BEGIN and END
        dataset = MorphDataset(bucket_inputs, bucket_targets, symbol_codes, target_symbol_codes, actual_bucket_length)
        batch_size = params.get("batch_size", 32)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        train_data_loaders.append(loader)

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RuMorphemeModel(params["model_params"], symbols, target_symbols)
    model.to(device)

    # Model parameters
    criterion = nn.CrossEntropyLoss(ignore_index=PAD)
    optimizer = optim.Adam(model.parameters())
    nepochs = params["model_params"].get("nepochs", 10)

    # Enabling logging to wandb
    wandb_enabled = params["wandb"].get("enabled", False)
    if wandb_enabled:
        wandb_project = params["wandb"].get("project", False)
        wandb.init(project=wandb_project, config=params)

    # Training loop
    for epoch in range(nepochs):
        total_loss = 0.0
        for dataloader in train_data_loaders:
            loss = train_model(model, dataloader, criterion, optimizer, device)
            total_loss += loss
        avg_loss = total_loss / len(train_data_loaders)
        if wandb_enabled:
            wandb.log({'loss': avg_loss})
        print(f"Эпоха {epoch + 1}/{nepochs}, Потеря: {avg_loss:.4f}")

    # Session completed
    if wandb_enabled:
        wandb.finish()

    # Check if directory exists
    if not os.path.exists('./model'):
        os.makedirs('./model')

    # Save model
    model_path = params.get("model_file", "model/pytorch_model.bin")
    torch.save(model.state_dict(), model_path)
    print(f"Модель сохранена в {model_path}")

    # Save model config
    config_file = params.get("config_file", "model/config.json")
    with open(config_file, "w") as f:
        json.dump(params["model_params"], f, indent=2, ensure_ascii=False)
    print(f"Конфигурация модели сохранена в {config_file}")

    # Save vocabulary
    vocab_path = params.get("vocab_file", "model/vocab.json")
    with open(vocab_path, "w") as f:
        json.dump({
            "symbols": symbols,
            "symbol_codes": symbol_codes,
        }, f, indent=2, ensure_ascii=False)
    print(f"Словарь сохранён в {vocab_path}")
