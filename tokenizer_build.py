import os
from tqdm import tqdm

from tokenizers import pre_tokenizers, Tokenizer, NormalizedString
from tokenizers.models import WordLevel

from rumorpheme.tokenizer import (
    RuMorphemeTokenizerFast, RuMorphemePreTokenizer,
    AUXILIARY, NUMBERS, UNKNOWN, BEGIN, END, PAD, CAP, ALL_CAPS,
    LETTERS_CYRILLIC, LETTERS_LATIN
)

DEFAULT_VOCAB = AUXILIARY + NUMBERS + LETTERS_CYRILLIC + LETTERS_LATIN


def build_morpheme_vocab(dataset, sort_vocab=True):
    vocab = {}
    # Assign IDs to DEFAULT_VOCAB tokens starting from 0, maintaining their order
    output = {k: idx for idx, k in enumerate(DEFAULT_VOCAB)}
    morpheme_set = set(output)  # To avoid duplicates
    morpheme_list = []  # To maintain insertion order if needed
    morpheme_id = len(DEFAULT_VOCAB)  # IDs for morphemes start after DEFAULT_VOCAB

    splitter = pre_tokenizers.Sequence([
        pre_tokenizers.Whitespace(),
        pre_tokenizers.Punctuation(),
        pre_tokenizers.Digits(individual_digits=True),
    ])
    pre_tokenizer = RuMorphemePreTokenizer(model_name="./model")

    for text in tqdm(dataset):
        words = splitter.pre_tokenize_str(text)
        for word in words:
            token_text = word[0]
            if token_text.isdigit() or token_text.isspace():
                # Skip digits and spaces already in DEFAULT_VOCAB
                if token_text.isspace() and token_text not in morpheme_set:
                    morpheme_set.add(token_text)
                    vocab[token_text] = morpheme_id
                    morpheme_list.append(token_text)
                    morpheme_id += 1
                continue

            # Tokenize each word by morpheme or by character for unknowns
            morphemes = pre_tokenizer.morpheme_or_char_split(0, NormalizedString(token_text))
            for morpheme in morphemes:
                morpheme_str = str(morpheme)
                if morpheme_str in {AUXILIARY[CAP], AUXILIARY[ALL_CAPS]}:
                    continue
                if morpheme_str not in morpheme_set:
                    morpheme_set.add(morpheme_str)
                    vocab[morpheme_str] = morpheme_id
                    morpheme_list.append(morpheme_str)
                    morpheme_id += 1

    # Now, sort morpheme tokens if required
    if sort_vocab:
        sorted_morphemes = sorted(morpheme_list)
    else:
        sorted_morphemes = morpheme_list  # Keep insertion order

    # Assign IDs to morpheme tokens, starting from len(DEFAULT_VOCAB)
    morpheme_vocab = {}
    for idx, token in enumerate(sorted_morphemes, start=len(DEFAULT_VOCAB)):
        morpheme_vocab[token] = idx

    # Combine DEFAULT_VOCAB and morpheme_vocab
    full_vocab = {**output, **morpheme_vocab}

    return full_vocab


# Example dataset for vocabulary creation
# dataset = [
#     "Аве, Цезарь!",
#     "Привет! Как твои дела?",
#     "Приветливей видали.",
#     "1912г.",
#     "Заманчиво! 123",
#     "Это тестовое предложение.",
#     "Морфемная токенизация на русском языке."
#     "Скажи-ка, дядя ведь не даром..."
# ]
dataset = open('./data/all_text.txt', 'r').read().splitlines()
# print(">>> ", len(dataset))
# exit()

# Build the vocabulary
morpheme_vocab = build_morpheme_vocab(dataset)
# print(">>> ", morpheme_vocab)
# exit()

# Create the WordLevel model with your vocabulary
model = WordLevel(vocab=morpheme_vocab, unk_token=AUXILIARY[UNKNOWN])

# Create the tokenizer with the model
tokenizer = Tokenizer(model=model)
# tokenizer.post_processor = processors.TemplateProcessing(
#     single=f"{AUXILIARY[BEGIN]}:0 $A:0 {AUXILIARY[END]}:0",
#     pair=f"{AUXILIARY[BEGIN]}:0 $A:0 {AUXILIARY[END]}:0 $B:1 {AUXILIARY[END]}:1",
#     special_tokens=[(f"{AUXILIARY[BEGIN]}", BEGIN), (f"{AUXILIARY[END]}", END)],
# )

# Check if directory exists
if not os.path.exists('tokenizer'):
    os.makedirs('tokenizer')

# Save the tokenizer to the directory
tokenizer.save('./tokenizer/tokenizer.json')

# Wrap it with RuMorphemeTokenizerFast for compatibility with transformers
new_tokenizer = RuMorphemeTokenizerFast(
    model_name="./model",
    tokenizer_object=tokenizer,
    # vocab=vocab,
    bos_token=AUXILIARY[BEGIN],
    eos_token=AUXILIARY[END],
    pad_token=AUXILIARY[PAD],
    unk_token=AUXILIARY[UNKNOWN],
)

# Test the tokenizer
# test_text = "Философское восприятие мира."
test_text = "Привет! Как твои дела?"
input_ids = new_tokenizer.encode(test_text)
print("Text:", test_text)
print("Encoded:", input_ids, len(input_ids))
print("Tokens:", new_tokenizer.convert_ids_to_tokens(input_ids))
print("Decoded:", new_tokenizer.decode(input_ids))

# Save the tokenizer for use with transformers
new_tokenizer.save_pretrained('./tokenizer')
