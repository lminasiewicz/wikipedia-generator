import torch
import re
from transformers import BertTokenizerFast
from functools import partial
from transformer_model import WikipediaGeneratorModel



def infer_with_model(prompt: str|None = None, model_path: str = "../models/transformer1.pth", cycles: int = 5, get_output: bool = False) -> str|None:
    if prompt is None:
        prompt = input("Please enter the prompt for the model to complete (The model can only remember around 100 words of context!):")
    prompt = prompt.strip()
    
    if bool(re.search(r'[^\x00-\x7F]', prompt)): print("Invalid Input. (Please only use ASCII symbols)")
    else:
        # instantiate tokenizer
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

        # load model
        model_dict = torch.load(model_path)
        model = WikipediaGeneratorModel(len(tokenizer), d_model=64, head_count=8, num_decoder_layers=6, dropout=0.1, max_len=128)
        model.load_state_dict(model_dict)
        print(f"Loaded Model: {model_path}")

        # prevent learning
        model.eval()
        
        output = ""
        for _ in range(cycles):
            # tokenize input
            prompt_tokenized = tokenizer.encode(prompt + " " + output)
            prompt_tokenized = torch.tensor(prompt_tokenized, dtype=torch.int32)
            
            # feed into the model and decode
            embedded_output = model(prompt_tokenized)
            tokenized_output = []

            for embedded_word in embedded_output:
                tokenized_output.append(torch.argmax(embedded_word).item())
            
            output += tokenizer.decode(tokenized_output).replace("[PAD]", "").strip() + " "

        if get_output:
            return output
        else:
            print("OUTPUT:")
            print(prompt + "\n\n" + output)


if __name__ == "__main__":
    infer_with_model()






