import argparse
import os
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import MarianMTModel, MarianTokenizer


def encode_batch(batch, tokenizer, model, layer_num, field="text"):

    tok_batch = tokenizer(
        batch[field],
        return_tensors="pt",
        padding="longest",
        return_attention_mask=True,
        truncation=True,
        max_length=128,
    )

    for k, v in tok_batch.items():
        tok_batch[k] = v.to(model.device)

    with torch.no_grad():
        enc_batch = model(**tok_batch, return_dict=True, output_hidden_states=True)

    sent_reps_curr_layer_mean = masked_mean(
        enc_batch.hidden_states[layer_num],
        tok_batch.attention_mask.unsqueeze(2).bool(),
        1,
    )

    out_dict = {f"mean_{layer_num}": sent_reps_curr_layer_mean.detach().cpu().numpy()}
    return out_dict


def masked_mean(
    vector: torch.Tensor, mask: torch.BoolTensor, dim: int, keepdim: bool = False
) -> torch.Tensor:
    def tiny_value_of_dtype(dtype: torch.dtype):
        if not dtype.is_floating_point:
            raise TypeError("Only supports floating point dtypes.")
        if dtype == torch.float or dtype == torch.double:
            return 1e-13
        elif dtype == torch.half:
            return 1e-4
        else:
            raise TypeError("Does not support dtype " + str(dtype))

    replaced_vector = vector.masked_fill(~mask, 0.0)
    value_sum = torch.sum(replaced_vector, dim=dim, keepdim=keepdim)
    value_count = torch.sum(mask, dim=dim, keepdim=keepdim)
    return value_sum / value_count.float().clamp(min=tiny_value_of_dtype(torch.float))


def extract_sent_reps(hf_model_dir, txt_dataset_path, batch_size, layer_num, gpu):
    model = MarianMTModel.from_pretrained(hf_model_dir).model.encoder
    model.eval()
    if gpu:
        model.cuda()
    tokenizer = MarianTokenizer.from_pretrained(hf_model_dir)
    dataset = load_dataset("text", data_files=txt_dataset_path, split="train")
    dataset = dataset.map(
        function=encode_batch,
        fn_kwargs={
            "tokenizer": tokenizer,
            "model": model,
            "layer_num": layer_num,
        },
        batched=True,
        batch_size=batch_size,
    )

    dataset = np.stack(dataset[f"mean_{layer_num}"], axis=0)
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-model-dir", type=str, help="Path to marian model file")
    parser.add_argument(
        "--txt-dataset-path", type=str, help="Path to marian decoder config"
    )
    parser.add_argument(
        "--batch-size", type=int, default=1000, help="Path to marian model file"
    )
    parser.add_argument(
        "--layer-num", type=int, default=4, help="Layer to extract embeddings from"
    )
    parser.add_argument(
        "--gpu",
        action='store_true',
        help="Whether to use GPU for batched dataset encoding",
    )
    parser.add_argument(
        "--out-filename", type=str, help="Path to file to save embeddings in npz format"
    )
    args = parser.parse_args()

    features = extract_sent_reps(
        hf_model_dir=args.hf_model_dir,
        txt_dataset_path=args.txt_dataset_path,
        batch_size=args.batch_size,
        layer_num=args.layer_num,
        gpu=args.gpu,
    )

    Path(os.path.dirname(args.out_filename)).mkdir(parents=True, exist_ok=True)
    np.savez(args.out_filename, features)
    print(f"Saved embeddings to {args.out_filename}")
