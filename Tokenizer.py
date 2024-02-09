from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
import json
from pathlib import Path
from torch.utils.data import random_split, Dataset, DataLoader
from Arabic_GP_Dataset import Arabic_GP_Dataset

DATASET_PATH = "Data\\Dataset.json"


def load_data(json_path):
    with open(json_path, "r", encoding="utf-8") as file:  # Open the file in read mode ,with utf-8 for arabic
        try:
            json_data = json.load(file)
            return json_data
        except json.JSONDecodeError as e:
            print(f"Error loading JSON: {e}")  # Handle potential JSON decoding errors


def get_all_words_or_phonetic(dataset, type):
    for word_and_phonetic in dataset:
        if type == "word":
            yield word_and_phonetic[0]
        elif type == "phonetic":
            yield word_and_phonetic[1]
        else:
            raise ValueError("Invalid type. Must be 'word' or 'phonetic'.")


def get_or_build_tokenizer(config, dataset, type):
    tokenizer_path = Path(config['tokenizer_file'].format(type))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=1)
        tokenizer.train_from_iterator(get_all_words_or_phonetic(dataset, type), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


def get_dataset(config):
    dataset_raw = load_data(DATASET_PATH)

    tokenizer_src = get_or_build_tokenizer(config, dataset_raw, config["src"])
    tokenizer_tgt = get_or_build_tokenizer(config, dataset_raw, config["tgt"])

    train_dataset_raw_size = int(0.9 * len(dataset_raw))
    val_dataset_raw_size = len(dataset_raw) - train_dataset_raw_size
    train_dataset_raw, val_dataset_raw = random_split(dataset_raw, [train_dataset_raw_size, val_dataset_raw_size])
    train_dataset = Arabic_GP_Dataset(train_dataset_raw, tokenizer_src, tokenizer_tgt, config["src"],
                                      config["tgt"], config["seq_len"])
    val_dataset = Arabic_GP_Dataset(val_dataset_raw, tokenizer_src, tokenizer_tgt, config["src"],
                                    config["tgt"], config["seq_len"])

    max_len_src = 0
    max_len_tgt = 0
    for item in dataset_raw:
        src_ids = tokenizer_src.encode(item[0]).ids
        tgt_ids = tokenizer_tgt.encode(item[1]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')

    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

