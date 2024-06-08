import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from transformer_model import WikipediaGeneratorModel
import os
import sys
sys.path.append("..")
from get_dataset import get_dataset


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DATASET_FILEPATH = "../dataset/wikipedia.txt"
BATCH_SIZE = 64
EPOCHS = 3
LEARNING_RATE = 1e-3
EMBED_SIZE = 128



def get_next_id() -> str:
    current_highest = 0
    for filepath in os.listdir("../models"):
        if filepath.startswith("transformer") and filepath.endswith(".pth"):
            id = int(filepath.removeprefix("transformer").removesuffix(".pth"))
            if id > current_highest:
                current_highest = id
    return f"transformer{current_highest + 1}.pth"



def main() -> None:
    print(f"\nUsing device: {DEVICE}\n")

    dataset, vocab_size = get_dataset(DATASET_FILEPATH)
    print(f"Dataset {DATASET_FILEPATH} loaded successfully.")
    dataset.set_format(type='torch', columns=['input_ids'])
    max_length = len(dataset["input_ids"][0])

    dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"Created DataLoader.")
    model = WikipediaGeneratorModel(vocab_size, d_model=EMBED_SIZE, head_count=8, num_decoder_layers=6, dropout=0.1, max_len=max_length).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()


    best_model = None
    best_loss = np.inf

    # Training
    model.train()
    for epoch in range(EPOCHS):
        for batch in dataloader:
            batch = batch["input_ids"].to(DEVICE)
            # x doesn't have last element, y doesn't have first element
            x_batch = batch[:, :-1]
            y_batch = batch[1:, :]

            loss = 0
            for x, y in zip(x_batch, y_batch):
                y_prediction = model(x)
                loss += loss_fn(y_prediction, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Validation (every epoch)
        model.eval()
        loss = 0
        with torch.no_grad():
            for batch in dataloader:
                batch = batch["input_ids"].to(DEVICE)
                x_batch = batch[:, :-1]
                y_batch = batch[1:, :]

                y_prediction = model(x_batch)
                loss += loss_fn(y_prediction, y_batch)
                
                if loss < best_loss:
                    best_loss = loss
                    best_model = model.state_dict()
        # print final state after each epoch
        print(f"Epoch {epoch}: Loss = {loss}")

    # Save the best model
    torch.save(best_model, get_next_id())



if __name__ == "__main__":
    main()