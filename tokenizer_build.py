import os

from tokenizers import pre_tokenizers, Tokenizer, NormalizedString
from tokenizers.models import WordLevel

from rumorpheme.tokenizer import (
    RuMorphemeTokenizerFast, RuMorphemePreTokenizer,
    AUXILIARY, NUMBERS, UNKNOWN, BEGIN, END
)


def build_morpheme_vocab(dataset, sort_vocab=True):
    vocab = {}
    output = {k: idx for idx, k in enumerate(AUXILIARY + NUMBERS)}
    morpheme_id = len(AUXILIARY + NUMBERS)
    splitter = pre_tokenizers.Sequence([
        pre_tokenizers.Whitespace(),
        pre_tokenizers.Punctuation(),
    ])
    pre_tokenizer = RuMorphemePreTokenizer()
    morpheme_list = []
    morpheme_set = set()
    for text in dataset:
        words = splitter.pre_tokenize_str(text)
        for word in words:
            token_text = word[0]
            if token_text.isspace():
                morpheme_str = token_text
                if morpheme_str not in morpheme_set:
                    morpheme_set.add(morpheme_str)
                    morpheme_list.append(morpheme_str)
            else:
                morphemes = pre_tokenizer.morpheme_split(0, NormalizedString(token_text))
                for morpheme in morphemes:
                    morpheme_str = str(morpheme)
                    if morpheme_str not in morpheme_set:
                        morpheme_set.add(morpheme_str)
                        morpheme_list.append(morpheme_str)

    if sort_vocab:
        sorted_morphemes = sorted(morpheme_list)
        for morpheme_str in sorted_morphemes:
            vocab[morpheme_str] = morpheme_id
            morpheme_id += 1
    else:
        for morpheme_str in morpheme_list:
            vocab[morpheme_str] = morpheme_id
            morpheme_id += 1

    return output | vocab


# Example dataset for vocabulary creation
dataset = [
    "Аве, Цезарь!",
    "Привет! Как твои дела?",
    "Приветливей видали.",
    "Заманчиво!",
    "Это тестовое предложение.",
    "Морфемная токенизация на русском языке."
    "Скажи-ка, дядя ведь не даром..."
]

# Build the vocabulary
vocab = build_morpheme_vocab(dataset)
# print(vocab)
# exit()

# Create the WordLevel model with your vocabulary
model = WordLevel(vocab=vocab, unk_token=AUXILIARY[UNKNOWN])

# Create the tokenizer with the model
tokenizer = Tokenizer(model=model)
# tokenizer.post_processor = processors.TemplateProcessing(
#     single=f"{AUXILIARY[BEGIN]}:0 $A:0 {AUXILIARY[END]}:0",
#     pair=f"{AUXILIARY[BEGIN]}:0 $A:0 {AUXILIARY[END]}:0 $B:1 {AUXILIARY[END]}:1",
#     special_tokens=[(f"{AUXILIARY[BEGIN]}", BEGIN), (f"{AUXILIARY[END]}", END)],
# )

# Check if directory exists
if not os.path.exists('./tokenizer'):
    os.makedirs('./tokenizer')

# Save the tokenizer to the directory
tokenizer.save('./tokenizer/tokenizer.json')

# Wrap it with RuMorphemeTokenizerFast for compatibility with transformers
new_tokenizer = RuMorphemeTokenizerFast(
    tokenizer_object=tokenizer,
    # vocab=vocab,
    bos_token=AUXILIARY[BEGIN],
    eos_token=AUXILIARY[END],
)

# Test the tokenizer
# test_text = "Философское восприятие мира."
test_text = "Привет! Как твои дела?"
input_ids = new_tokenizer.encode(test_text)
print("Text: ", test_text)
print("Encoded:", input_ids)
print("Tokens:", new_tokenizer.convert_ids_to_tokens(input_ids))
print("Decoded:", new_tokenizer.decode(input_ids))

# Save the tokenizer for use with transformers
new_tokenizer.save_pretrained('./tokenizer')
