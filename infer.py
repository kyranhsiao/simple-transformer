import torch
from transformers import AutoTokenizer, AutoConfig
from model.simple_transformer import Transformer


def test_infer():
    base_model = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    config = AutoConfig.from_pretrained(base_model)
    transformer = Transformer(config)


    ckpt_path = ""
    ckpt_dict = torch.load(ckpt_path, map_location="cpu")
    transformer.load_state_dict(ckpt_dict["model_state_dict"])
    transformer.eval()

    src_txt = "time flies like an arrow"
    src_input = tokenizer(src_txt, return_tensors="pt", add_special_tokens=False)
    src_ids = src_input["input_ids"]

    batch_size = src_ids.size(0)
    start_token_id = None
    max_len = 20

    if start_token_id is None:
        start_token_id = tokenizer.cls_token_id or tokenizer.pad_token_id or 101
    
    tgt_ids = torch.full((batch_size, 1), start_token_id, dtype=torch.long, device=src_ids.device)

    with torch.no_grad():
        for _ in range(max_len):
            logits = transformer(src_ids, tgt_ids,
                                return_logits=True)
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            tgt_ids = torch.cat([tgt_ids, next_token], dim=1)

            if (next_token == tokenizer.sep_token_id).any():
                break
    decoded_txt = tokenizer.batch_decode(tgt_ids, skip_special_tokens=True)
    print(f"{'decoded text: ':<20}{decoded_txt}")

if __name__ == "__main__":
    test_infer()

