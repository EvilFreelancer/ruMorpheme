from transformers import AutoTokenizer

# Wrap it with RuMorphemeTokenizerFast for compatibility with transformers
tokenizer = AutoTokenizer.from_pretrained("./tokenizer", trust_remote_code=True)

test_text = "Философское восприятие мира."
# test_text = "Привет! Как твои дела?"
input_ids = tokenizer.encode(test_text)

print("Text:", test_text)
print("Encoded:", input_ids)
print("Tokens:", tokenizer.convert_ids_to_tokens(input_ids))
print("Decoded:", tokenizer.decode(input_ids))
