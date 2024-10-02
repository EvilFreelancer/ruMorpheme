import numpy as np
import torch
import torch.nn as nn

# Constants
PAD, BEGIN, END, UNKNOWN = 0, 1, 2, 3
AUXILIARY = ['PAD', 'BEGIN', 'END', 'UNKNOWN']


class Partitioner(nn.Module):
    def __init__(self, symbols_number, target_symbols_number, params):
        super(Partitioner, self).__init__()
        self.symbols_number = symbols_number
        self.target_symbols_number = target_symbols_number

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
        x = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        x = x.permute(0, 2, 1)  # [batch_size, embedding_dim, seq_len]
        conv_outputs = []
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


def prepare_data(data, symbol_codes, bucket_length):
    batch_size = len(data)
    inputs = np.full((batch_size, bucket_length), PAD, dtype=int)
    inputs[:, 0] = BEGIN
    for i, word in enumerate(data):
        word_codes = [symbol_codes.get(char, UNKNOWN) for char in word]
        inputs[i, 1:1 + len(word_codes)] = word_codes
        inputs[i, 1 + len(word_codes)] = END
    return inputs


def labels_to_morphemes(word, labels, log_probs, target_symbols, use_morpheme_types):
    morphemes = []
    morpheme_types = []
    morpheme_log_probs = []
    curr_morpheme = ""
    curr_morpheme_log_probs = []
    prev_label_type = None  # Initialize prev_label_type
    if use_morpheme_types:
        end_labels = ['E-ROOT', 'E-PREF', 'E-SUFF', 'E-END', 'E-POSTFIX', 'S-ROOT',
                      'S-PREF', 'S-SUFF', 'S-END', 'S-LINK', 'S-HYPH']
    else:
        end_labels = ['E-None', 'S-None']
    for i, (letter, label_idx) in enumerate(zip(word, labels)):
        label = target_symbols[label_idx]
        morpheme_type = label.split('-')[-1] if '-' in label else label
        if letter == '-':
            # Save current morpheme if any
            if curr_morpheme:
                morphemes.append(curr_morpheme)
                morpheme_types.append(prev_label_type if prev_label_type else 'UNKNOWN')
                avg_log_prob = sum(curr_morpheme_log_probs) / len(curr_morpheme_log_probs)
                morpheme_log_probs.append(avg_log_prob)
                curr_morpheme = ""
                curr_morpheme_log_probs = []
            # Treat hyphen as a separate morpheme of type HYPH
            morphemes.append(letter)
            morpheme_types.append('HYPH')
            morpheme_log_probs.append(log_probs[i][label_idx])
            prev_label_type = 'HYPH'
        else:
            curr_morpheme += letter
            curr_morpheme_log_probs.append(log_probs[i][label_idx])
            if label in end_labels:
                morphemes.append(curr_morpheme)
                morpheme_types.append(morpheme_type)
                avg_log_prob = sum(curr_morpheme_log_probs) / len(curr_morpheme_log_probs)
                morpheme_log_probs.append(avg_log_prob)
                curr_morpheme = ""
                curr_morpheme_log_probs = []
                prev_label_type = morpheme_type
    # Process any remaining morpheme
    if curr_morpheme:
        morphemes.append(curr_morpheme)
        morpheme_types.append(prev_label_type if prev_label_type else 'UNKNOWN')
        avg_log_prob = sum(curr_morpheme_log_probs) / len(curr_morpheme_log_probs)
        morpheme_log_probs.append(avg_log_prob)
    morpheme_probs = [np.exp(lp) * 100 for lp in morpheme_log_probs]
    return morphemes, morpheme_types, morpheme_probs
