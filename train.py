import argparse
import os
from datetime import datetime
from transformers import AutoTokenizer, AutoConfig
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from model.simple_transformer import Transformer
from utils import *

def test_train():
    model_ckpt = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    config = AutoConfig.from_pretrained(model_ckpt)
    transformer = Transformer(config)
    transformer.train()

    lr = 1e-4
    optimizer = optim.Adam(transformer.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    src_txt = "time flies like an arrow"
    tgt_txt = "fast birds fly"

    src_input = tokenizer(src_txt, return_tensors="pt", add_special_tokens=True)
    tgt_input = tokenizer(tgt_txt, return_tensors="pt", add_special_tokens=True)

    src_ids = src_input["input_ids"]
    tgt_ids = tgt_input["input_ids"]

    dec_input_ids = tgt_ids[:, :-1]
    dec_true_ids = tgt_ids[:, 1:]

    logits = transformer(src_ids, dec_input_ids, 
                         return_logits=True)
    
    logits = logits.reshape(-1, logits.size(-1))
    dec_true_ids = dec_true_ids.reshape(-1)

    loss = loss_fn(logits, dec_true_ids)
    print(f"{'Loss: ':<20}{loss.item():.6f}")

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

def train_loop(model, loss_fn, optimizer, src_ids, tgt_ids):
    model.train()
    dec_input_ids = tgt_ids[:, :-1]
    dec_true_ids = tgt_ids[:, 1:]

    logits = model(src_ids, dec_input_ids, 
                         return_logits=True)
    logits = logits.reshape(-1, logits.size(-1))

    dec_true_ids = dec_true_ids.reshape(-1)

    loss = loss_fn(logits, dec_true_ids)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return loss, logits

def eval_loop(model, tokenizer, src_ids, tgt_ids, max_len):
    model.eval()
    with torch.no_grad():
        for _ in range(max_len):
            logits = model(src_ids, tgt_ids,
                                return_logits=True)
            next_tokens_logits = logits[:, -1, :]
            next_tokens = torch.argmax(next_tokens_logits, dim=-1, keepdim=True)
            tgt_ids = torch.cat([tgt_ids, next_tokens], dim=1)

            if (next_tokens == tokenizer.sep_token_id).any():
                break
        pred_txts = tokenizer.batch_decode(tgt_ids, skip_special_tokens=True)

    return pred_txts
        

def train(args):
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    config = AutoConfig.from_pretrained(args.base_model)
    transformer = Transformer(config)
    transformer.train()

    train_dataset = load_dataset(args.train_dataset_path)
    eval_dataset = load_dataset(args.eval_dataset_path)
    optimizer = optim.Adam(transformer.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in tqdm(range(1, args.num_epochs + 1), desc=f"Training epochs", leave=True):
        total_loss = 0.0
        for i in range(0, len(train_dataset), args.batch_size):
            batch_data = train_dataset[i:i+args.batch_size]
            src_txts = [sample["zh"] for sample in batch_data]
            tgt_txts = [sample["en"] for sample in batch_data]
            src_inputs = tokenizer(src_txts, return_tensors="pt",
                                   padding=True, truncation=True, 
                                   add_special_tokens=True)
            tgt_inputs = tokenizer(tgt_txts, return_tensors="pt", 
                                   padding=True, truncation=True, 
                                   add_special_tokens=True)

            src_ids = src_inputs["input_ids"]
            tgt_ids = tgt_inputs["input_ids"]
            loss, _ = train_loop(transformer, loss_fn, optimizer,
                       src_ids, tgt_ids)
            total_loss += loss.item()
            # logger.info(f"{'Loss: ':<10}{loss.item():.6f}")
        
        epoch_loss = total_loss / len(train_dataset) / args.batch_size
        logger.info(f"Epoch [{epoch:>4}/{args.num_epochs}] loss: {epoch_loss:.6f}")

        if epoch % args.eval_interval == 0:
            eval(transformer, tokenizer, eval_dataset, args.batch_size)
        
        if epoch % args.save_interval == 0:
            ckpt_path = os.path.join(args.ckpt_dir, f"{args.base_model}_{epoch}.pt")
            state_dict = {
                "epoch": epoch,
                "lr": args.lr,
                "base_model": args.base_model,
                "optimizer_name": "adam",
                "loss_fn": "cross_entropy_loss",
                "model_state_dict": transformer.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            save_ckpt(state_dict, ckpt_path)
            logger.info(f"save checkpoint to {ckpt_path}")

def eval(model, tokenizer, dataset, batch_size, start_token_id=None, max_len=10):

    for i in range(0, len(dataset), batch_size):
        batch_data = dataset[i:i+batch_size]

        src_txts = [sample["zh"] for sample in batch_data]
        src_inputs = tokenizer(src_txts, return_tensors="pt", 
                               padding=True, truncation=True,
                               add_special_tokens=False)
        src_ids = src_inputs["input_ids"]
        batch_size = src_ids.size(0)
        if start_token_id is None:
            start_token_id = tokenizer.cls_token_id or tokenizer.pad_token_id or 101
        tgt_ids = torch.full((batch_size, 1), start_token_id, dtype=torch.long, device=src_ids.device)

        pred_txts = eval_loop(model, tokenizer, src_ids, tgt_ids, max_len=max_len)
        logger.info(f"{src_txts}{'-'*10}{pred_txts}")


if __name__ == "__main__":
    # test_train()
    parser = argparse.ArgumentParser(description="Demo to train `simple transformer`")
    parser.add_argument('--base_model', type=str, default="bert-base-multilingual-cased",
                        help="Pretrained model name or path")
    parser.add_argument('--ckpt_dir', type=str, default="checkpoints",
                        help="directory to save checkpoints")

    parser.add_argument('--train_dataset_path', type=str, default="dataset/zh-en_train.jsonl",
                        help="path to training dataset")
    parser.add_argument('--eval_dataset_path', type=str, default="dataset/zh-en_eval.jsonl",
                        help="path to evaluation dataset")
    parser.add_argument('--log_dir', type=str, default="log", 
                        help="directory to log file")

    parser.add_argument('--num_epochs', type=int, default=100, 
                        help="number of training epoch")
    parser.add_argument('--eval_interval', type=int, default=1, 
                        help="interval of evaluation")
    parser.add_argument('--save_interval', type=int, default=1, 
                        help="interval of saving")
    parser.add_argument('--batch_size', type=int, default=2, 
                        help="batch size")
    parser.add_argument('--lr', type=float, default=1e-4, 
                        help="learning rate")
    
    args = parser.parse_args()

    log_file = f'train-{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.log'
    init_logger(os.path.join(args.log_dir, log_file))
    train(args)


