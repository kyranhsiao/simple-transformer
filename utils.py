import torch
import json
import logging

logger = logging.getLogger("train-logger")

def load_dataset(file_path):
    data_list = []

    with open(file_path, mode='r', encoding="utf-8") as reader:
        for line in reader:
            data_dict = json.loads(line.strip())
            data_list.append(data_dict)
    
    return data_list

def save_ckpt(state_dict, file_path):
    ckpt = state_dict
    torch.save(ckpt, file_path)

def init_logger(file_path, level=logging.INFO):
    logger.setLevel(level)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        file_handler = logging.FileHandler(file_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
            