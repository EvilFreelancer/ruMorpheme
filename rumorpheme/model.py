import sys
import os
import json
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

from .utils import prepare_data
from .const import TARGET_SYMBOLS


class RuMorphemeModel(nn.Module):
    def __init__(self, params, symbols, symbol_codes=None):
        super(RuMorphemeModel, self).__init__()

        # Available symbols
        self.symbols = symbols
        self.symbols_number = len(self.symbols)
        self.symbol_codes = symbol_codes
        if symbol_codes is None:
            self.symbol_codes = {a: i for i, a in enumerate(symbols)}

        # Target symbols
        self.target_symbols = TARGET_SYMBOLS
        self.target_symbols_number = len(self.target_symbols)
        self.target_symbol_codes = {a: i for i, a in enumerate(TARGET_SYMBOLS)}

        # Extract parameters from config
        self.use_morpheme_types = params.get('use_morpheme_types', True)
        self.measure_last = params.get("measure_last", self.use_morpheme_types)
        self.embeddings_size = params.get('embeddings_size', 32)
        self.conv_layers = params.get('conv_layers', 1)
        self.window_size = params.get('window_size', [5])
        self.filters_number = params.get('filters_number', 64)
        self.dense_output_units = params.get('dense_output_units', 0)
        self.use_lstm = params.get('use_lstm', False)
        self.lstm_units = params.get('lstm_units', 64)
        self.dropout = params.get('dropout', 0.0)
        self.context_dropout = params.get('context_dropout', 0.0)
        self.models_number = params.get('models_number', 1)

        # Ensure window_size and filters_number are lists
        if isinstance(self.window_size, int):
            self.window_size = [self.window_size]
        if isinstance(self.filters_number, int):
            self.filters_number = [self.filters_number] * len(self.window_size)

        # Define layers
        self.embedding = nn.Embedding(self.symbols_number, self.embeddings_size)
        self.conv_layers_list = nn.ModuleList()
        input_channels = self.embeddings_size
        for i in range(self.conv_layers):
            convs = nn.ModuleList()
            for ws, fn in zip(self.window_size, self.filters_number):
                conv = nn.Conv1d(input_channels, fn, ws, padding=ws // 2)
                convs.append(conv)
            self.conv_layers_list.append(convs)
            input_channels = sum(self.filters_number)
        if self.use_lstm:
            self.lstm = nn.LSTM(input_channels, self.lstm_units, batch_first=True, bidirectional=True)
            lstm_output_size = self.lstm_units * 2
            fc_input_dim = lstm_output_size
        else:
            fc_input_dim = input_channels
        if self.dense_output_units > 0:
            self.fc1 = nn.Linear(fc_input_dim, self.dense_output_units)
            self.fc_out = nn.Linear(self.dense_output_units, self.target_symbols_number)
        else:
            self.fc_out = nn.Linear(fc_input_dim, self.target_symbols_number)
        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self, x):
        """
        Function to perform forward pass of the model.
        :param x:
        :return:
        """
        x = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        x = x.permute(0, 2, 1)  # [batch_size, embedding_dim, seq_len]
        for convs in self.conv_layers_list:
            conv_outs = []
            for conv in convs:
                conv_out = conv(x)
                conv_out = torch.relu(conv_out)
                conv_outs.append(conv_out)
            x = torch.cat(conv_outs, dim=1)
            x = self.dropout_layer(x)
        x = x.permute(0, 2, 1)  # [batch_size, seq_len, channels]
        if self.use_lstm:
            x, _ = self.lstm(x)
        if self.dense_output_units > 0:
            x = torch.relu(self.fc1(x))
        logits = self.fc_out(x)  # [batch_size, seq_len, target_symbols_number]
        return logits

    @property
    def device(self):
        """Returns the device of the model."""
        return next(self.parameters()).device

    @classmethod
    def from_pretrained(cls, model_path, device=None):
        # Let's use cuda if available
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if os.path.isdir(model_path):
            # Load files from local directory
            config_file = os.path.join(model_path, "config.json")
            vocab_file = os.path.join(model_path, "vocab.json")
            model_file = os.path.join(model_path, "pytorch_model.bin")
        else:
            # Assume model_path is a Hugging Face repo ID
            repo_id = model_path
            config_file = hf_hub_download(repo_id=repo_id, filename="config.json")
            vocab_file = hf_hub_download(repo_id=repo_id, filename="vocab.json")
            model_file = hf_hub_download(repo_id=repo_id, filename="pytorch_model.bin")

        if not all(os.path.isfile(f) for f in [config_file, vocab_file, model_file]):
            print("Model files could not to be read.")
            sys.exit(1)

        # Read parameters of model
        with open(config_file, "r", encoding="utf8") as fin:
            params = json.load(fin)

        # Load vocabularies
        with open(vocab_file, "rb") as f:
            vocab_data = json.load(f)
        symbols = vocab_data["symbols"]
        symbol_codes = vocab_data["symbol_codes"]

        # Load model
        model = RuMorphemeModel(params, symbols, symbol_codes)
        model.load_state_dict(torch.load(model_file, map_location=device, weights_only=True))
        return model

    def predict(self, data: list):
        """
        Function to perform predictions of a word.
        :param data: it should be a list of words (or a list with a single word)
        :return:
        """
        # Disable gradient calculation
        with torch.no_grad():
            input_data = prepare_data(data, self.symbol_codes)
            inputs = torch.tensor(input_data, dtype=torch.long).to(self.device)  # Move data to same device as model
            outputs = self.forward(inputs)
            log_probs = torch.log_softmax(outputs, dim=-1)
            predictions = torch.argmax(log_probs, dim=-1)

        # Return predictions copied to CPU memory
        return predictions.cpu().numpy(), log_probs.cpu().numpy()
