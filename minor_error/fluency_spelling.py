import argparse
import random
import nltk
from nltk import word_tokenize
from tqdm import tqdm

nltk.download('punkt')

# List of additional symbols and characters from various language families
latin_characters = [chr(i) for i in range(0x0041, 0x007A)]  # Latin
cyrillic_characters = [chr(i) for i in range(0x0400, 0x04FF)]  # Cyrillic
arabic_characters = [chr(i) for i in range(0x0600, 0x06FF)]  # Arabic
chinese_characters = [chr(i) for i in range(0x4E00, 0x9FFF)]  # Chinese
japanese_characters = [chr(i) for i in range(0x3040, 0x30FF)]  # Japanese (Hiragana and Katakana)
devanagari_characters = [chr(i) for i in range(0x0900, 0x097F)]  # Devanagari

# Combine all character lists into one list
foreign_symbols = latin_characters + cyrillic_characters + arabic_characters + chinese_characters + japanese_characters + devanagari_characters

def random_case_change(word):
    """Randomly change the case of letters in a word."""
    return ''.join(random.choice([char.upper(), char.lower()]) for char in word)

def delete_whitespace(tokens):
    """Randomly delete a whitespace between words."""
    if len(tokens) > 1:
        index = random.randint(0, len(tokens) - 2)
        tokens[index] = tokens[index] + tokens[index + 1]
        del tokens[index + 1]
    return tokens

def create_incorrect_spelling(word):
    """Create an incorrect spelling by switching characters, changing characters, or adding foreign symbols."""
    if len(word) > 2:
        action = random.choice(['switch', 'change', 'insert'])
        if action == 'switch':
            index = random.randint(0, len(word) - 2)
            word = word[:index] + word[index + 1] + word[index] + word[index + 2:]
        elif action == 'change':
            index = random.randint(0, len(word) - 1)
            new_char = random.choice('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')
            word = word[:index] + new_char + word[index + 1:]
        elif action == 'insert':
            index = random.randint(0, len(word))
            new_char = random.choice(foreign_symbols)
            word = word[:index] + new_char + word[index:]
    return word

def modify_spelling(tokens):
    """Randomly modify the spelling of a token."""
    if not tokens:
        return tokens, False

    index = random.randint(0, len(tokens) - 1)
    word = tokens[index]

    # Set probabilities for each type of modification
    modification_type = random.choices(
        ['case', 'whitespace', 'incorrect'],
        weights=[0.4, 0.2, 0.4],
        k=1
    )[0]

    if modification_type == 'case':
        tokens[index] = random_case_change(word)
    elif modification_type == 'whitespace':
        tokens = delete_whitespace(tokens)
    elif modification_type == 'incorrect':
        tokens[index] = create_incorrect_spelling(word)

    return tokens, True

def process_line(line):
    line = line.strip()
    parts = line.strip().split('\t')
    if len(parts) != 3:
        return None  # Indicate that the line should be skipped

    tokenized = word_tokenize(parts[1])

    # Skip lines with no tokenized words
    if not tokenized:
        return None  # Indicate that the line should be skipped

    # Modify spelling in the tokenized second element
    modified_tokenized, modified = modify_spelling(tokenized)

    if modified:
        modified_second_element = ' '.join(modified_tokenized).strip()
        third_element = str(float(parts[2]) - 1.0)
        modified_line = f"{parts[0]}\t{modified_second_element}\t{third_element}"
        return modified_line

    return None  # Indicate that the line should be skipped

def process_file(input_file, output_file):
    # Count the number of lines in the input file for tqdm progress bar
    with open(input_file, 'r') as infile:
        total_lines = sum(1 for _ in infile)

    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in tqdm(infile, total=total_lines, desc="Processing lines"):
            modified_line = process_line(line)
            if modified_line is not None:
                outfile.write(modified_line + '\n')

def main():
    parser = argparse.ArgumentParser(description="Modify text in a file.")
    parser.add_argument("input_file", help="Path to the input file")
    parser.add_argument("output_file", help="Path to the output file")
    args = parser.parse_args()

    process_file(args.input_file, args.output_file)

if __name__ == "__main__":
    main()
