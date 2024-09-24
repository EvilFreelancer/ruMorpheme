# чтение и разметка данных
import numpy as np


def generate_BMES(morphs, morph_types):
    answer = []
    for morph, morph_type in zip(morphs, morph_types):
        if len(morph) == 1:
            answer.append("S-" + morph_type)
        else:
            answer.append("B-" + morph_type)
            answer.extend(["M-" + morph_type] * (len(morph) - 2))
            answer.append("E-" + morph_type)
    return answer


def read_splitted(infile, transform_to_BMES=True, n=None, morph_sep="/", shuffle=True):
    source, targets = [], []
    with open(infile, "r", encoding="utf8") as fin:
        for line in fin:
            line = line.strip()
            if line == "":
                break
            word, analysis = line.split("\t")
            morphs = analysis.split(morph_sep)
            morph_types = ["None"] * len(morphs)
            if transform_to_BMES:
                target = generate_BMES(morphs, morph_types)
            else:
                target = morph_types
            source.append(word)
            targets.append(target)
    indexes = list(range(len(source)))
    if shuffle:
        np.random.shuffle(indexes)
    if n is not None:
        indexes = indexes[:n]
    source = [source[i] for i in indexes]
    targets = [targets[i] for i in indexes]
    return source, targets


def read_BMES(infile, transform_to_BMES=True, n=None,
              morph_sep="/" ,sep=":", shuffle=True):
    source, targets = [], []
    with open(infile, "r", encoding="utf8") as fin:
        for line in fin:
            line = line.strip()
            if line == "":
                break
            word, analysis = line.split("\t")
            analysis = [x.split(sep) for x in analysis.split(morph_sep)]
            morphs, morph_types = [elem[0] for elem in analysis], [elem[1] for elem in analysis]
            target = generate_BMES(morphs, morph_types) if transform_to_BMES else morphs
            source.append(word)
            targets.append(target)
    indexes = list(range(len(source)))
    if shuffle:
        np.random.shuffle(indexes)
    if n is not None:
        indexes = indexes[:n]
    source = [source[i] for i in indexes]
    targets = [targets[i] for i in indexes]
    return source, targets
