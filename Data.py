import json

JSON_PATH = "JsonData\\ar.json"


def load_data(json_path):
    with open(json_path, "r", encoding="utf-8") as file:  # Open the file in read mode ,with utf-8 for arabic
        try:
            json_data = json.load(file)  # Load the JSON data from the file
            parsed_data = json_data["ar"]  # Access the "ar" key within the data
            return parsed_data[0]
        except json.JSONDecodeError as e:
            print(f"Error loading JSON: {e}")  # Handle potential JSON decoding errors


def _remove_backslash(pronunciation):
    return pronunciation[1:-1]


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


if __name__ == '__main__':
    data_with_duplicates = load_data(JSON_PATH)
    data_without_duplicates = clean_duplicates(data_with_duplicates)
    data_without_backslash = remove_backslash(data_without_duplicates)
