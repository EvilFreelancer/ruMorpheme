import os
import json
import re
import string
from typing import List

from tokenizers import pre_tokenizers, decoders, NormalizedString, PreTokenizedString, AddedToken
from transformers import PreTrainedTokenizerFast

from rumorpheme import RuMorphemeModel, labels_to_morphemes

DEFAULT_MODEL_NAME = "evilfreelancer/ruMorpheme-v0.2"

END, BEGIN, PAD, UNKNOWN, CAP, ALL_CAPS = 0, 1, 2, 3, 4, 5
SYSTEM, USER, ASSISTANT, FUNCTION_CALL, FUNCTION_RESPONSE = 6, 7, 8, 9, 10
SPACE, NEWLINE, TAB = 11, 12, 13

AUXILIARY = [
    "</s>", "<s>", "<pad>", "<unk>", "<cap>", "<all_caps>",
    "system", "user", "assistant", "function_call", "function_response",
    " ", "\n", "\t"
]

NUMBERS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
LETTERS_CYRILLIC = list(map(chr, range(ord('а'), ord('я') + 1)))
LETTERS_LATIN = list(string.ascii_lowercase)


class RuMorphemePreTokenizer:
    """
    Pre-tokenizer for RuMorpheme model.
    Splits on spaces, newlines, and tabs, including these as tokens.
    Then, applies morpheme splitting to non-space tokens.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        self.model = RuMorphemeModel.from_pretrained(model_name)
        self.model.eval()

    def pre_tokenize(self, pretok: PreTokenizedString):
        # First, split on spaces (including newlines and tabs) and add them as tokens
        pretok.split(self.split_on_spaces)

        # Apply morpheme or character-level splitting to non-space tokens
        pretok.split(self.morpheme_or_char_split)

    def split_on_spaces(self, i: int, normalized_string: NormalizedString) -> List[NormalizedString]:
        """
        Splits on spaces, newlines, and tabs, including these as tokens.
        """
        text = str(normalized_string)
        splits = [
            NormalizedString(match.group())
            for match in re.finditer(r'\s+|\S+', text)
        ]

        # Convert newlines and tabs to tokens
        for idx, split in enumerate(splits):
            if split == "\n":
                splits[idx] = NormalizedString(AUXILIARY[NEWLINE])
            elif split == "\t":
                splits[idx] = NormalizedString(AUXILIARY[TAB])

        return splits

    def morpheme_or_char_split(self, i: int, normalized_string: NormalizedString) -> List[NormalizedString]:
        """
        Attempts to split the token into morphemes. If the token starts with "UNKNOWN/",
        splits it into individual characters.
        """
        word = str(normalized_string)

        # If the token is whitespace or digits, return as is
        if word.isspace() or word.isdigit():
            return [normalized_string]

        # Ignore tokens that are only punctuation or non-alphabetical
        if not any(c.isalpha() for c in word):
            return [normalized_string]

        # Detect capitalization and add relevant token if necessary
        cap_token = None
        if word[0].isupper():
            cap_token = NormalizedString(AUXILIARY[CAP])
            if len(word) > 1 and word.isupper():
                cap_token = NormalizedString(AUXILIARY[ALL_CAPS])

        # Convert word to lowercase for morpheme splitting
        word_lower = word.lower()

        # Make predictions to get morphemes
        all_predictions, all_log_probs = self.model.predict([word_lower])
        morphs, morph_types, _ = labels_to_morphemes(word_lower, all_predictions[0], all_log_probs[0])

        # Handle unknown tokens by splitting into characters
        morpheme_tokens = []
        for morph, morph_type in zip(morphs, morph_types):
            if morph_type == "UNKNOWN":
                # Split unknown morpheme into characters
                char_tokens = [NormalizedString(char) for char in morph]
                morpheme_tokens.extend(char_tokens)
            else:
                # Add as a single morpheme token
                morpheme_tokens.append(NormalizedString(f"{morph_type}/{morph}"))

        # Insert capitalization token if needed
        if cap_token:
            return [cap_token] + morpheme_tokens
        else:
            return morpheme_tokens


class RuMorphemeDecoder:
    """
    Custom decoder for RuMorpheme model, it removes morph_type prefix from tokens and keeps spaces.
    """

    def decode_chain(self, tokens: List[str]) -> List[str]:
        """
        tokenizer.decode function calls this function
        """
        decoded_tokens = []
        capitalize_next = False
        uppercase_next = False

        for token in tokens:
            # Handle capitalization tokens
            if token == AUXILIARY[CAP]:
                capitalize_next = True
                continue
            elif token == AUXILIARY[ALL_CAPS]:
                uppercase_next = True
                continue

            # If token is a space, keep it as is
            if token.isspace():
                decoded_tokens.append(token)
            else:
                # Remove morph_type prefix if present
                if '/' in token:
                    _, morph = token.split('/', 1)
                else:
                    morph = token

                # Apply capitalization if needed
                if uppercase_next:
                    morph = morph.upper()
                    uppercase_next = False
                elif capitalize_next:
                    morph = morph.capitalize()
                    capitalize_next = False

                decoded_tokens.append(morph)
        return decoded_tokens


class RuMorphemeTokenizerFast(PreTrainedTokenizerFast):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # If pre-tokenizer nodel is not specified, use the default
        self.model_name = kwargs.get('model_name')
        if kwargs.get('model_name') is None:
            self.model_name: str = DEFAULT_MODEL_NAME

        # Complete initialization
        self.init_backend_tokenizer()

    def init_backend_tokenizer(self):
        # Custom pre-tokenizer
        self.backend_tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Punctuation(),
            pre_tokenizers.Digits(individual_digits=True),
            pre_tokenizers.PreTokenizer.custom(RuMorphemePreTokenizer(self.model_name))
        ])
        # Custom decoder
        self.backend_tokenizer.decoder = decoders.Decoder.custom(RuMorphemeDecoder())

    def save_pretrained(self, save_directory, **kwargs):
        # Temporarily remove the custom pre-tokenizer and decoder before saving
        original_pre_tokenizer = self.backend_tokenizer.pre_tokenizer
        original_decoder = self.backend_tokenizer.decoder
        self.backend_tokenizer.pre_tokenizer = None
        self.backend_tokenizer.decoder = None

        # Save the tokenizer using the parent method
        super().save_pretrained(save_directory, **kwargs)

        # Re-attach the custom pre-tokenizer and decoder
        self.backend_tokenizer.pre_tokenizer = original_pre_tokenizer
        self.backend_tokenizer.decoder = original_decoder

        # Save the tokenizer class name in tokenizer_config.json
        tokenizer_config_file = os.path.join(save_directory, 'tokenizer_config.json')
        if os.path.isfile(tokenizer_config_file):
            with open(tokenizer_config_file, 'r', encoding='utf-8') as f:
                tokenizer_config = json.load(f)
        else:
            tokenizer_config = {}

        # Correctly specify the tokenizer_class with module name
        tokenizer_config['tokenizer_class'] = "RuMorphemeTokenizerFast"
        tokenizer_config['use_fast'] = True
        tokenizer_config['auto_map'] = {"AutoTokenizer": ["", "tokenizer.RuMorphemeTokenizerFast"]}

        with open(tokenizer_config_file, 'w', encoding='utf-8') as f:
            json.dump(tokenizer_config, f, ensure_ascii=False)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *init_inputs, **kwargs):
        # Load the tokenizer using the parent method
        tokenizer = super(RuMorphemeTokenizerFast, cls).from_pretrained(
            pretrained_model_name_or_path, *init_inputs, **kwargs
        )

        # If pre-tokenizer nodel is not specified, use the default
        model_name = kwargs.get('model_name')
        if kwargs.get('model_name') is None:
            model_name: str = DEFAULT_MODEL_NAME

        # Custom pre-tokenizer
        tokenizer.backend_tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Punctuation(),
            pre_tokenizers.Digits(individual_digits=True),
            pre_tokenizers.PreTokenizer.custom(RuMorphemePreTokenizer(model_name))
        ])

        # Custom decoder
        tokenizer.backend_tokenizer.decoder = decoders.Decoder.custom(RuMorphemeDecoder())

        return tokenizer
