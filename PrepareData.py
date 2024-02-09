import json
import sqlite3

AR_JSON_PATH = "Data\\ar.json"
SYMBOLS_JSON_PATH = "Data\\symbols.json"
DATABASE_PATH = "Data\\lexicon.db"
DATASET_PATH = "Data\\Dataset.json"


def load_data(json_path):
    with open(json_path, "r", encoding="utf-8") as file:  # Open the file in read mode ,with utf-8 for arabic
        try:
            json_data = json.load(file)  # Load the JSON data from the file
            parsed_data = json_data["ar"]  # Access the "ar" key within the data
            return parsed_data[0]
        except json.JSONDecodeError as e:
            print(f"Error loading JSON: {e}")  # Handle potential JSON decoding errors


def _remove_backslash(phoneme):
    return phoneme[1:-1]


def _remove_white_space(phoneme):
    return phoneme.replace(" ", "")


def _add_space_between_letters(string):
    return ' '.join(string)


def remove_backslash(data):
    for word, pronunciations in data.items():
        data[word] = _remove_backslash(pronunciations)
    print(list(data.items())[:50])
    return data


def clean_duplicates(data):
    # Iterate through the original dictionary
    for word, pronunciations in data.items():
        # Check if there are multiple pronunciations
        if len(pronunciations.split(", ")) > 1:
            # Split the pronunciations and keep the first one
            pronunciations_list = pronunciations.split(", ")
            data[word] = pronunciations_list[0]  # Update with the first pronunciation
    return data


def fetch_word_phonemes(database_path):
    word_phonemes_dict = {}

    # Connect to the SQLite database
    conn = sqlite3.connect(database_path)
    c = conn.cursor()
    # Fetch data from the table
    c.execute("SELECT word, phonemes FROM word_phonemes")
    rows = c.fetchall()
    # Iterate over the rows and populate the dictionary
    for row in rows:
        word_phonemes_dict[row[0]] = row[1]
    # Close the connection
    conn.close()
    return word_phonemes_dict


def remove_white_space(data):
    for word, phonemes in data.items():
        data[word] = _remove_white_space(phonemes)
    print(list(data.items())[:50])
    return data


def get_all_symbols(dataset):
    word_symbols = set()
    phoneme_symbols = set()

    for word, phoneme in dataset.items():
        word_symbols.update(word)
        phoneme_symbols.update(phoneme)

    symbols = {
        "word symbols": sorted(word_symbols),
        "phoneme symbols": sorted(phoneme_symbols)
    }

    with open(SYMBOLS_JSON_PATH, "w", encoding="utf-8") as json_file:
        json.dump(symbols, json_file, ensure_ascii=False, indent=4)
    for word, phonemes in dataset.items():
        dataset[word] = _remove_white_space(phonemes)
    print(list(dataset.items())[:50])
    return dataset


def add_space_between_letters(dataset):
    dataset_letters = {}
    for word, phonemes in dataset.items():
        dataset_letters[_add_space_between_letters(word)] = _add_space_between_letters(phonemes)
    print(list(dataset_letters.items())[:50])
    return dataset_letters


if __name__ == '__main__':
    json_data_with_duplicates = load_data(AR_JSON_PATH)
    json_data_without_duplicates = clean_duplicates(json_data_with_duplicates)
    json_data_clean = remove_backslash(json_data_without_duplicates)
    database_data_with_space = fetch_word_phonemes(DATABASE_PATH)
    database_data_clean = remove_white_space(database_data_with_space)
    dataset_words = {**json_data_clean, **database_data_clean}
    get_all_symbols(dataset_words)
    dataset = add_space_between_letters(dataset_words)
    with open(DATASET_PATH, 'w', encoding="utf-8") as json_file:
        json.dump(list(dataset.items()), json_file, ensure_ascii=False, indent=4)
    print("dataset Prepared")
