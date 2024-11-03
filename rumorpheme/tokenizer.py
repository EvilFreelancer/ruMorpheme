import os
import json
import re
from typing import List

from tokenizers import pre_tokenizers, decoders, NormalizedString, PreTokenizedString
from transformers import PreTrainedTokenizerFast

from rumorpheme import RuMorphemeModel, labels_to_morphemes

DEFAULT_MODEL_NAME = "evilfreelancer/ruMorpheme-v0.1"
PAD, BEGIN, END, UNKNOWN, SPACE, SYSTEM, USER, ASSISTANT, FUNCTION_CALL, FUNCTION_RESPONSE = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
AUXILIARY = ["<pad>", "<s>", "</s>", "<unk>", " ", "system", "user", "assistant", "function_call", "function_response"]
NUMBERS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]


class RuMorphemePreTokenizer:
    """
    Pre-tokenizer for RuMorpheme model.
    Splits on spaces and includes spaces as tokens.
    Then, applies morpheme splitting to non-space tokens.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        self.model = RuMorphemeModel.from_pretrained(model_name)
        self.model.eval()

    def pre_tokenize(self, pretok: PreTokenizedString):
        # First, split on spaces and include spaces as tokens
        pretok.split(self.split_on_spaces)
        # Then, apply morpheme splitting to non-space tokens
        pretok.split(self.morpheme_split)

    def split_on_spaces(self, i: int, normalized_string: NormalizedString) -> List[NormalizedString]:
        """
        Splits on spaces and includes spaces as tokens.
        TODO: Need to make performance tests on this function.
        """
        text = str(normalized_string)
        splits = [NormalizedString(match.group()) for match in re.finditer(r'\s+|\S+', text)]
        return splits

    def morpheme_split(self, i: int, normalized_string: NormalizedString) -> List[NormalizedString]:
        """
        Split word on morphemes, including numbers and punctuation.
        """
        word = str(normalized_string)

        # If word is just spaces, return as is
        if word.isspace():
            return [normalized_string]

        # Ignore special characters (non-alphabetical and non-numeric)
        if not any(c.isalpha() or c.isdigit() for c in word):
            return [normalized_string]

        # Make predictions and return morphemes
        all_predictions, all_log_probs = self.model.predict([word])
        morphs, morph_types, _ = labels_to_morphemes(word.lower(), all_predictions[0], all_log_probs[0])
        return [NormalizedString(f"{morph_type}/{morph}") for morph, morph_type in zip(morphs, morph_types)]


class RuMorphemeDecoder:
    """
    Custom decoder for RuMorpheme model, it removes morph_type prefix from tokens and keep spaces.
    """

    def decode_chain(self, tokens: List[str]) -> List[str]:
        """
        tokenizer.decode function calls this function
        """
        decoded_tokens = []
        for token in tokens:
            # If token is a space, keep it as is
            if token.isspace():
                decoded_tokens.append(token)
            else:
                # Remove morph_type prefix if present
                if '/' in token:
                    _, morph = token.split('/', 1)
                else:
                    morph = token
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
        tokenizer_config['auto_map'] = {"AutoTokenizer": ["", "my_tokenizer.RuMorphemeTokenizerFast"]}

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
            pre_tokenizers.PreTokenizer.custom(RuMorphemePreTokenizer(model_name))
        ])

        # Custom decoder
        tokenizer.backend_tokenizer.decoder = decoders.Decoder.custom(RuMorphemeDecoder())

        return tokenizer
