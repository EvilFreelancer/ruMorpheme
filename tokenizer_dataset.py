import re
import pandas as pd

path = "./data/dostoevsky.csv"
dataframe = pd.read_csv(path)
dataframe = dataframe.dropna(subset=['text'])
texts = dataframe['text'].tolist()
all_text = ' '.join(texts)
processed_text = re.sub(' {2,}', '\n', all_text)

open('./data/all_text.txt', 'w').write(processed_text)
