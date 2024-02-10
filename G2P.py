from pathlib import Path
import json
import torch
from tokenizers import Tokenizer

from Model import build_transformer
from config import get_config, latest_weights_file_path

SEARCHSPACEPATH = "Data/SearchSpace.json"


def __add_space_between_letters(string):
    return ' '.join(string)


def __remove_white_space(phoneme):
    return phoneme.replace(" ", "")


def load_data(json_path):
    with open(json_path, "r", encoding="utf-8") as file:  # Open the file in read mode ,with utf-8 for arabic
        try:
            json_data = json.load(file)  # Load the JSON data from the file
            return json_data
        except json.JSONDecodeError as e:
            print(f"Error loading JSON: {e}")  # Handle potential JSON decoding errors


def __Arabic_G2P(word: str, device, model, tokenizer_src, tokenizer_tgt, seq_len):
    sentence = __add_space_between_letters(word)
    # translate the sentence
    model.eval()
    with torch.no_grad():
        # Precompute the encoder output and reuse it for every generation step
        source = tokenizer_src.encode(sentence)
        source = torch.cat([
            torch.tensor([tokenizer_src.token_to_id('[SOS]')], dtype=torch.int64),
            torch.tensor(source.ids, dtype=torch.int64),
            torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype=torch.int64),
            torch.tensor([tokenizer_src.token_to_id('[PAD]')] * (seq_len - len(source.ids) - 2), dtype=torch.int64)
        ], dim=0).to(device)
        source_mask = (source != tokenizer_src.token_to_id('[PAD]')).unsqueeze(0).unsqueeze(0).int().to(device)
        encoder_output = model.encode(source, source_mask)

        # Initialize the decoder input with the sos token
        decoder_input = torch.empty(1, 1).fill_(tokenizer_tgt.token_to_id('[SOS]')).type_as(source).to(device)
        # Generate the translation word by word
        while decoder_input.size(1) < seq_len:
            # build mask for target and calculate output
            decoder_mask = torch.triu(torch.ones((1, decoder_input.size(1), decoder_input.size(1))), diagonal=1).type(
                torch.int).type_as(source_mask).to(device)
            out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

            # project next token
            prob = model.project(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            decoder_input = torch.cat(
                [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1)

            # print the translated word
            # print(f"{tokenizer_tgt.decode([next_word.item()])}", end=' ')

            # break if we predict the end of sentence token
            if next_word == tokenizer_tgt.token_to_id('[EOS]'):
                break

        # convert ids to tokens
    return tokenizer_tgt.decode(decoder_input[0].tolist())


def Arabic_G2P(sentence: str):
    # load Search Space
    SearchSpace = load_data(SEARCHSPACEPATH)
    # Define the device, tokenizers, and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'using device : {device}')
    config = get_config()
    tokenizer_src = Tokenizer.from_file(str(Path(config['tokenizer_file'].format(config['src']))))
    tokenizer_tgt = Tokenizer.from_file(str(Path(config['tokenizer_file'].format(config['tgt']))))
    model = build_transformer(tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size(), config["seq_len"],
                              config['seq_len'], d_model=config['d_model']).to(device)
    # Load the pretrained weights
    model_filename = latest_weights_file_path(config)
    state = torch.load(model_filename)
    model.load_state_dict(state['model_state_dict'])
    seq_len = config['seq_len']

    # process words
    words = sentence.split()
    Phoneme_results = []
    # Retrieve the first key
    first_key = next(iter(SearchSpace))

    # Print the first key
    for word in words:
        if word in SearchSpace:
            Phoneme_results.append(SearchSpace[word])
        else:
            Phoneme_results.append(
                __remove_white_space(__Arabic_G2P(word, device, model, tokenizer_src, tokenizer_tgt, seq_len)))
    return ' '.join(Phoneme_results)


if __name__ == "__main__":
    print(Arabic_G2P('هذا النص هو مثال'))
