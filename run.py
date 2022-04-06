from argparse import ArgumentParser
import pandas as pd
import numpy as np
from pytest import param
import torch
import os
import random
from torch import optim

from src.data.dataset import EmbeddingVector
from src.utils import config
from src.modules.transform import AutoEncoder
from src.utils import engine

parser = ArgumentParser()
parser.add_argument("--data_dir", type=str, default="/data")
parser.add_argument("--output_dir", type=str, default="/output")
parser.add_argument("--model_path", type=str, required=False)
parser.add_argument("--reduced_output_path", type=str, required=False)
parser.add_argument("--train_size", type=float, default=0.85)
parser.add_argument("--train_batch_size", type=int, default=64)
parser.add_argument("--val_batch_size", type=int, default=256)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--lr", type=float, default=1e-2)

parser.add_argument("--encode", action="store_true")

parser.add_argument("--seed", type=int, default=42)

args = parser.parse_args()

def set_seed(seed = args.seed):
    torch.random.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def run(params, save_model=True):
    set_seed()

    text_encodings = np.load(os.path.join(args.data_dir, "encoding.npy"))
    bio_encodings = np.load(os.path.join(args.data_dir, "bio_encoding.npy"))
    bio_ids = pd.read_csv(os.path.join(args.data_dir, "bio_id.csv"))
    thoughts = pd.read_csv(os.path.join(args.data_dir, "thoughts.tsv"), sep="\t")

    device = config.DEVICE
    model = AutoEncoder(config.INPUT_SIZE, config.OUTPUT_SIZE)

    if args.encode:
        model.load_state_dict(torch.load(args.model_path))
        model.to(device)

        dataset = EmbeddingVector(
            text_encodings,
            bio_encodings,
            bio_ids,
            thoughts
        )

        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=args.val_batch_size
        )

        reduced_encodings = engine.encode_fn(data_loader, model, device)
        np.save(args.reduced_output_path, reduced_encodings)

    else:
        model.to(device)
        thoughts_train = thoughts.sample(frac=args.train_size)
        thoughts_test = thoughts.drop(thoughts_train.index)

        train_dataset = EmbeddingVector(
            text_encodings, 
            bio_encodings,
            bio_ids,
            thoughts_train
        )
        train_data_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.train_batch_size
        )

        test_dataset = EmbeddingVector(
            text_encodings, 
            bio_encodings,
            bio_ids,
            thoughts_test
        )
        test_data_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.val_batch_size
        )
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.001,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = optim.Adam(optimizer_parameters, lr=params['lr'])

        early_stopping_iter = 3
        early_stopping_counter = 0

        best_loss = np.inf
        for epoch in range(args.epochs):
            train_loss = engine.train_fn(train_data_loader, model, optimizer, device)
            test_loss = engine.eval_fn(test_data_loader, model, device)
            if test_loss < best_loss:
                best_loss = test_loss
                if save_model:
                    torch.save(model.state_dict(), os.path.join(args.output_dir, f'{epoch}_model.bin'))
            else:
                early_stopping_counter += 1

            if early_stopping_iter < early_stopping_counter:
                break
            
            print(f"EPOCH[{epoch+1}]: train loss: {train_loss}, test loss: {test_loss}, best loss: {best_loss}")

        return best_loss

def main():
    params = {
        "lr": args.lr,
    }

    run(params)

if __name__=="__main__":
    main()