import numpy as np

from rumorpheme.const import TARGET_SYMBOLS


def labels_to_morphemes(word: str, labels, log_probs, use_morpheme_types=True):
    """
    Function for labeling predicted labels due to target symbols.
    :param word:
    :param labels:
    :param log_probs:
    :param use_morpheme_types:
    :return:
    """
    morphemes = []
    morpheme_types = []
    morpheme_log_probs = []
    curr_morpheme = ""
    curr_morpheme_log_probs = []
    prev_label_type = None  # Initialize prev_label_type
    if use_morpheme_types:
        end_labels = ['E-ROOT', 'E-PREF', 'E-SUFF', 'E-END', 'E-POSTFIX',
                      'S-ROOT', 'S-PREF', 'S-SUFF', 'S-END', 'S-LINK', 'S-HYPH']
    else:
        end_labels = ['E-None', 'S-None']
    for i, (letter, label_idx) in enumerate(zip(word, labels)):
        label = TARGET_SYMBOLS[label_idx]
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
