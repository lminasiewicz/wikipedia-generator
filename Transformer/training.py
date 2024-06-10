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


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET_FILEPATH = "../dataset/wikipedia_small.txt"
BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 1e-3
EMBED_SIZE = 64



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
    dataset_length = len(dataset)
    num_batches = dataset_length // BATCH_SIZE

    dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"Created DataLoader.")
    model = WikipediaGeneratorModel(vocab_size, d_model=EMBED_SIZE, head_count=8, num_decoder_layers=6, dropout=0.1, max_len=max_length).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss().to(DEVICE)

    best_model = None
    best_loss = np.inf

    # Training
    print("Initializing Training...")
    model.train()
    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}")
        print("Training:")
        for i, batch in enumerate(dataloader, 0):
            sys.stdout.write(f"\r{i}/{num_batches}")
            batch = batch["input_ids"].to(DEVICE)
            # x doesn't have last element, y doesn't have first element
            x_batch = batch[:, :-1]
            y_batch = batch[:, 1:]

            loss = 0
            for x, y in zip(x_batch, y_batch):
                y_prediction = model(x)
                loss += loss_fn(y_prediction, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print()
        # # Validation (every epoch)
        # model.eval()
        # loss = 0
        # print("Training:")
        # with torch.no_grad():
        #     for i, batch in enumerate(dataloader, 0):
        #         sys.stdout.write(f"\r{i}/{num_batches}")
        #         batch = batch["input_ids"].to(DEVICE)
        #         x_batch = batch[:, :-1]
        #         y_batch = batch[1:, :]

        #         y_prediction = model(x_batch)
        #         loss += loss_fn(y_prediction, y_batch)
                
    #         if loss < best_loss:
    #             best_loss = loss
    #             best_model = model.state_dict()
        # # print final state after each epoch
        # print(f"Epoch {epoch}: Loss = {loss}")

    # ------------ WARNING: FOREGOING VALIDATION TO TRAIN THE MODEL IN A REASONABLE AMOUNT OF TIME ------------
    # Save the best model
    best_model = model.state_dict() # remove when validation is uncommented
    torch.save(best_model, f"../models/{get_next_id()}")



if __name__ == "__main__":
    main()