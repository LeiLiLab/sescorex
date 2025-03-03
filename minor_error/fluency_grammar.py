import argparse
import random
import nltk
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from tqdm import tqdm
import inflect

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()
p = inflect.engine()

def get_wordnet_pos(treebank_tag):
    """Map POS tag to first character lemmatize() accepts"""
    if treebank_tag.startswith('J'):
        return wn.ADJ
    elif treebank_tag.startswith('V'):
        return wn.VERB
    elif treebank_tag.startswith('N'):
        return wn.NOUN
    elif treebank_tag.startswith('R'):
        return wn.ADV
    else:
        return None

def modify_word(word, pos):
    """Modify the word by changing its tense or plurality"""
    wordnet_pos = get_wordnet_pos(pos) or wordnet.NOUN
    lemma = lemmatizer.lemmatize(word, pos=wordnet_pos)
    if wordnet_pos == wordnet.NOUN:
        if p.singular_noun(word):  # Check if the word is plural
            return lemma  # Convert plural to singular
        else:
            return p.plural(lemma)  # Convert singular to plural
    elif wordnet_pos == wordnet.VERB:
        # Randomly choose a tense modification
        tense_choice = random.choice(['past', 'present', 'future'])
        if tense_choice == 'past':
            return lemma + 'ed' if not lemma.endswith('e') else lemma + 'd'
        elif tense_choice == 'present':
            return lemma + 'ing' if not lemma.endswith('e') else lemma + 'ing'
        elif tense_choice == 'future':
            return 'will ' + lemma
    return word

def is_modifiable(tag):
    """Check if the tag represents a modifiable word (noun or verb)"""
    wordnet_pos = get_wordnet_pos(tag)
    return wordnet_pos in {wn.NOUN, wn.VERB}

def process_line(line):
    line = line.strip()
    parts = line.strip().split('\t')
    if len(parts) != 3:
        return line

    tokenized = word_tokenize(parts[1])
    tagged = pos_tag(tokenized)

    if not tagged:
        return line

    third_element_value = float(parts[2])
    num_modifications = 1
    if len(tagged) > 1 and third_element_value > -15 and random.random() < 0.4:
        num_modifications = random.randint(1, 3)

    modified_indices = set()
    modified = False

    for _ in range(num_modifications):
        # Find a new token to modify that hasn't been modified yet and is modifiable
        possible_indices = [i for i, (_, tag) in enumerate(tagged) if i not in modified_indices and is_modifiable(tag)]
        if not possible_indices:
            break
        random_token_index = random.choice(possible_indices)
        word, pos = tagged[random_token_index]
        modified_word = modify_word(word, pos)

        if word != modified_word:
            tokenized[random_token_index] = modified_word
            modified_indices.add(random_token_index)
            modified = True

    if modified:
        modified_second_element = ''.join([' ' + t if i > 0 and (t not in ",.!?\"':;)" and tokenized[i-1] not in "(\"") else t for i, t in enumerate(tokenized)])
        modified_second_element = modified_second_element.replace(" ' ", "'").replace(" ’ ", "’").replace(" ’s", "’s")
        third_element = str(third_element_value - len(modified_indices))
        modified_line = f"{parts[0]}\t{modified_second_element.strip()}\t{third_element}"
        return modified_line

    return line

def process_file(input_file, output_file):
    # Count the number of lines in the input file for tqdm progress bar
    with open(input_file, 'r') as infile:
        total_lines = sum(1 for _ in infile)

    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in tqdm(infile, total=total_lines, desc="Processing lines"):
            modified_line = process_line(line)
            outfile.write(modified_line + '\n')

def main():
    parser = argparse.ArgumentParser(description="Modify text in a file.")
    parser.add_argument("input_file", help="Path to the input file")
    parser.add_argument("output_file", help="Path to the output file")
    args = parser.parse_args()

    process_file(args.input_file, args.output_file)

if __name__ == "__main__":
    main()
