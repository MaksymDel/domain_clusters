# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# adapted from https://github.com/huggingface/transformers/blob/main/src/transformers/models/marian/convert_marian_to_pytorch.py

import argparse
import json
import os
import shutil
import warnings
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import torch
import yaml
from torch import nn
from transformers import MarianConfig, MarianMTModel, MarianTokenizer


def remove_suffix(text: str, suffix: str):
    if text.endswith(suffix):
        return text[: -len(suffix)]
    return text  # or whatever


def remove_prefix(text: str, prefix: str):
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text  # or whatever


def convert_encoder_layer(opus_dict, layer_prefix: str, converter: dict):
    sd = {}
    for k in opus_dict:
        if not k.startswith(layer_prefix):
            continue
        stripped = remove_prefix(k, layer_prefix)
        v = opus_dict[k].T  # besides embeddings, everything must be transposed.
        sd[converter[stripped]] = torch.tensor(v).squeeze()
    return sd


def load_layers_(
    layer_lst: nn.ModuleList, opus_state: dict, converter, is_decoder=False
):
    for i, layer in enumerate(layer_lst):
        layer_tag = f"decoder_l{i + 1}_" if is_decoder else f"encoder_l{i + 1}_"
        sd = convert_encoder_layer(opus_state, layer_tag, converter)
        layer.load_state_dict(sd, strict=False)


def add_emb_entries(wemb, final_bias, n_special_tokens=1):
    vsize, d_model = wemb.shape
    embs_to_add = np.zeros((n_special_tokens, d_model))
    new_embs = np.concatenate([wemb, embs_to_add])
    bias_to_add = np.zeros((n_special_tokens, 1))
    new_bias = np.concatenate((final_bias, bias_to_add), axis=1)
    return new_embs, new_bias


def _cast_yaml_str(v):
    bool_dct = {"true": True, "false": False}
    if not isinstance(v, str):
        return v
    elif v in bool_dct:
        return bool_dct[v]
    try:
        return int(v)
    except (TypeError, ValueError):
        return v


def cast_marian_config(raw_cfg: Dict[str, str]) -> Dict:
    return {k: _cast_yaml_str(v) for k, v in raw_cfg.items()}


CONFIG_KEY = "special:model.yml"


def load_config_from_state_dict(opus_dict):
    cfg_str = "".join([chr(x) for x in opus_dict[CONFIG_KEY]])
    yaml_cfg = yaml.load(cfg_str[:-1], Loader=yaml.BaseLoader)
    return cast_marian_config(yaml_cfg)


def add_to_vocab_(vocab: Dict[str, int], special_tokens: List[str]):
    start = max(vocab.values()) + 1
    added = 0
    for tok in special_tokens:
        if tok in vocab:
            continue
        vocab[tok] = start + added
        added += 1
    return added


def load_vocab(vocab_path):
    vocab = {}
    with open(vocab_path) as f:
        for i, l in enumerate(f):
            t, _ = l.split()
            vocab[t] = i
    return vocab


def check_equal(marian_cfg, k1, k2):
    v1, v2 = marian_cfg[k1], marian_cfg[k2]
    if v1 != v2:
        raise ValueError(f"hparams {k1},{k2} differ: {v1} != {v2}")


def check_marian_cfg_assumptions(marian_cfg):
    assumed_settings = {
        "layer-normalization": False,
        "right-left": False,
        "transformer-ffn-depth": 2,
        "transformer-aan-depth": 2,
        "transformer-no-projection": False,
        "transformer-postprocess-emb": "d",
        "transformer-postprocess": "dan",  # Dropout, add, normalize
        "transformer-preprocess": "",
        "type": "transformer",
        "ulr-dim-emb": 0,
        "dec-cell-base-depth": 2,
        "dec-cell-high-depth": 1,
        "transformer-aan-nogate": False,
    }
    for k, v in assumed_settings.items():
        actual = marian_cfg[k]
        if actual != v:
            raise ValueError(
                f"Unexpected config value for {k} expected {v} got {actual}"
            )


BIAS_KEY = "decoder_ff_logit_out_b"
BART_CONVERTER = {  # for each encoder and decoder layer
    "self_Wq": "self_attn.q_proj.weight",
    "self_Wk": "self_attn.k_proj.weight",
    "self_Wv": "self_attn.v_proj.weight",
    "self_Wo": "self_attn.out_proj.weight",
    "self_bq": "self_attn.q_proj.bias",
    "self_bk": "self_attn.k_proj.bias",
    "self_bv": "self_attn.v_proj.bias",
    "self_bo": "self_attn.out_proj.bias",
    "self_Wo_ln_scale": "self_attn_layer_norm.weight",
    "self_Wo_ln_bias": "self_attn_layer_norm.bias",
    "ffn_W1": "fc1.weight",
    "ffn_b1": "fc1.bias",
    "ffn_W2": "fc2.weight",
    "ffn_b2": "fc2.bias",
    "ffn_ffn_ln_scale": "final_layer_norm.weight",
    "ffn_ffn_ln_bias": "final_layer_norm.bias",
    # Decoder Cross Attention
    "context_Wk": "encoder_attn.k_proj.weight",
    "context_Wo": "encoder_attn.out_proj.weight",
    "context_Wq": "encoder_attn.q_proj.weight",
    "context_Wv": "encoder_attn.v_proj.weight",
    "context_bk": "encoder_attn.k_proj.bias",
    "context_bo": "encoder_attn.out_proj.bias",
    "context_bq": "encoder_attn.q_proj.bias",
    "context_bv": "encoder_attn.v_proj.bias",
    "context_Wo_ln_scale": "encoder_attn_layer_norm.weight",
    "context_Wo_ln_bias": "encoder_attn_layer_norm.bias",
}


class OpusState:
    def __init__(self, npz_model_path, yml_decoder_path, tokenizer):
        self.state_dict = np.load(npz_model_path)
        cfg = load_config_from_state_dict(self.state_dict)
        if cfg["dim-vocabs"][0] != cfg["dim-vocabs"][1]:
            raise ValueError
        if "Wpos" in self.state_dict:
            raise ValueError("Wpos key in state dictionary")
        self.state_dict = dict(self.state_dict)
        if cfg["tied-embeddings-all"]:
            cfg["tied-embeddings-src"] = True
            cfg["tied-embeddings"] = True
        self.share_encoder_decoder_embeddings = cfg["tied-embeddings-src"]

        # retrieve EOS token and set correctly
        tokenizer_has_eos_token_id = (
            hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None
        )
        eos_token_id = tokenizer.eos_token_id if tokenizer_has_eos_token_id else 0

        if cfg["tied-embeddings-src"]:
            self.wemb, self.final_bias = add_emb_entries(
                self.state_dict["Wemb"], self.state_dict[BIAS_KEY], 1
            )
            self.pad_token_id = self.wemb.shape[0] - 1
            cfg["vocab_size"] = self.pad_token_id + 1
        else:
            self.wemb, _ = add_emb_entries(
                self.state_dict["encoder_Wemb"], self.state_dict[BIAS_KEY], 1
            )
            self.dec_wemb, self.final_bias = add_emb_entries(
                self.state_dict["decoder_Wemb"], self.state_dict[BIAS_KEY], 1
            )
            # still assuming that vocab size is same for encoder and decoder
            self.pad_token_id = self.wemb.shape[0] - 1
            cfg["vocab_size"] = self.pad_token_id + 1
            cfg["decoder_vocab_size"] = self.pad_token_id + 1

        if cfg["vocab_size"] != tokenizer.vocab_size:
            raise ValueError(
                f"Original vocab size {cfg['vocab_size']} and new vocab size {len(tokenizer.encoder)} mismatched."
            )

        # self.state_dict['Wemb'].sha
        self.state_keys = list(self.state_dict.keys())
        if "Wtype" in self.state_dict:
            raise ValueError("Wtype key in state dictionary")
        self._check_layer_entries()
        self.cfg = cfg
        hidden_size, intermediate_shape = self.state_dict["encoder_l1_ffn_W1"].shape
        if hidden_size != cfg["dim-emb"]:
            raise ValueError(
                f"Hidden size {hidden_size} and configured size {cfg['dim_emb']} mismatched"
            )

        # Process decoder.yml
        decoder_yml = cast_marian_config(load_yaml(yml_decoder_path))
        check_marian_cfg_assumptions(cfg)
        self.hf_config = MarianConfig(
            vocab_size=cfg["vocab_size"],
            decoder_vocab_size=cfg.get("decoder_vocab_size", cfg["vocab_size"]),
            share_encoder_decoder_embeddings=cfg["tied-embeddings-src"],
            decoder_layers=cfg["dec-depth"],
            encoder_layers=cfg["enc-depth"],
            decoder_attention_heads=cfg["transformer-heads"],
            encoder_attention_heads=cfg["transformer-heads"],
            decoder_ffn_dim=cfg["transformer-dim-ffn"],
            encoder_ffn_dim=cfg["transformer-dim-ffn"],
            d_model=cfg["dim-emb"],
            activation_function=cfg["transformer-ffn-activation"],
            pad_token_id=self.pad_token_id,
            eos_token_id=eos_token_id,
            forced_eos_token_id=eos_token_id,
            bos_token_id=0,
            max_position_embeddings=cfg["dim-emb"],
            scale_embedding=True,
            normalize_embedding="n" in cfg["transformer-preprocess"],
            static_position_embeddings=not cfg["transformer-train-position-embeddings"],
            tie_word_embeddings=cfg["tied-embeddings"],
            dropout=0.1,  # see opus-mt-train repo/transformer-dropout param.
            # default: add_final_layer_norm=False,
            num_beams=decoder_yml["beam-size"],
            decoder_start_token_id=self.pad_token_id,
            bad_words_ids=[[self.pad_token_id]],
            max_length=512,
        )

    def _check_layer_entries(self):
        self.encoder_l1 = self.sub_keys("encoder_l1")
        self.decoder_l1 = self.sub_keys("decoder_l1")
        self.decoder_l2 = self.sub_keys("decoder_l2")
        if len(self.encoder_l1) != 16:
            warnings.warn(
                f"Expected 16 keys for each encoder layer, got {len(self.encoder_l1)}"
            )
        if len(self.decoder_l1) != 26:
            warnings.warn(
                f"Expected 26 keys for each decoder layer, got {len(self.decoder_l1)}"
            )
        if len(self.decoder_l2) != 26:
            warnings.warn(
                f"Expected 26 keys for each decoder layer, got {len(self.decoder_l1)}"
            )

    @property
    def extra_keys(self):
        extra = []
        for k in self.state_keys:
            if (
                k.startswith("encoder_l")
                or k.startswith("decoder_l")
                or k
                in [
                    CONFIG_KEY,
                    "Wemb",
                    "encoder_Wemb",
                    "decoder_Wemb",
                    "Wpos",
                    "decoder_ff_logit_out_b",
                ]
            ):
                continue
            else:
                extra.append(k)
        return extra

    def sub_keys(self, layer_prefix):
        return [
            remove_prefix(k, layer_prefix)
            for k in self.state_dict
            if k.startswith(layer_prefix)
        ]

    def load_marian_model(self) -> MarianMTModel:
        state_dict, cfg = self.state_dict, self.hf_config

        if not cfg.static_position_embeddings:
            raise ValueError("config.static_position_embeddings should be True")
        model = MarianMTModel(cfg)

        if "hidden_size" in cfg.to_dict():
            raise ValueError("hidden_size is in config")
        load_layers_(
            model.model.encoder.layers,
            state_dict,
            BART_CONVERTER,
        )
        load_layers_(
            model.model.decoder.layers, state_dict, BART_CONVERTER, is_decoder=True
        )

        # handle tensors not associated with layers
        if self.cfg["tied-embeddings-src"]:
            wemb_tensor = nn.Parameter(torch.FloatTensor(self.wemb))
            bias_tensor = nn.Parameter(torch.FloatTensor(self.final_bias))
            model.model.shared.weight = wemb_tensor
            model.model.encoder.embed_tokens = (
                model.model.decoder.embed_tokens
            ) = model.model.shared
        else:
            wemb_tensor = nn.Parameter(torch.FloatTensor(self.wemb))
            model.model.encoder.embed_tokens.weight = wemb_tensor

            decoder_wemb_tensor = nn.Parameter(torch.FloatTensor(self.dec_wemb))
            bias_tensor = nn.Parameter(torch.FloatTensor(self.final_bias))
            model.model.decoder.embed_tokens.weight = decoder_wemb_tensor

        model.final_logits_bias = bias_tensor

        if "Wpos" in state_dict:
            print("Unexpected: got Wpos")
            wpos_tensor = torch.tensor(state_dict["Wpos"])
            model.model.encoder.embed_positions.weight = wpos_tensor
            model.model.decoder.embed_positions.weight = wpos_tensor

        if cfg.normalize_embedding:
            if not ("encoder_emb_ln_scale_pre" in state_dict):
                raise ValueError("encoder_emb_ln_scale_pre is not in state dictionary")
            raise NotImplementedError("Need to convert layernorm_embedding")

        if self.extra_keys:
            raise ValueError(f"Failed to convert {self.extra_keys}")

        if model.get_input_embeddings().padding_idx != self.pad_token_id:
            raise ValueError(
                f"Padding tokens {model.get_input_embeddings().padding_idx} and {self.pad_token_id} mismatched"
            )
        return model


def load_yaml(path):
    with open(path) as f:
        return yaml.load(f, Loader=yaml.BaseLoader)


def save_json(content: Union[Dict, List], path: str) -> None:
    with open(path, "w") as f:
        json.dump(content, f)


def convert_tokenizer(
    vocab_path, spm_model_path, tgt_vocab_path=None, tgt_spm_model_path=None
):
    # ------------- setup -------------
    is_vocab_separate = False
    if tgt_vocab_path is not None:
        assert vocab_path != tgt_vocab_path
        is_vocab_separate = True

    vocab_dir = os.path.dirname(vocab_path)
    spm_dir = os.path.dirname(spm_model_path)
    assert vocab_dir == spm_dir
    temp_dir = Path(f"{vocab_dir}/tmp")
    temp_dir.mkdir()

    # ------------- convert spm -------------
    shutil.copyfile(spm_model_path, f"{temp_dir}/source.spm")

    if tgt_spm_model_path is None:
        shutil.copyfile(spm_model_path, f"{temp_dir}/target.spm")
    else:
        shutil.copyfile(tgt_spm_model_path, f"{temp_dir}/target.spm")

    # ------------- convert vocab -------------
    vocab = load_vocab(vocab_path)
    num_added = add_to_vocab_(vocab, ["<pad>"])
    print(f"added {num_added} tokens to vocab")
    save_json(vocab, temp_dir / "vocab.json")

    if tgt_vocab_path is not None:
        tgt_vocab = load_vocab(tgt_vocab_path)
        num_added = add_to_vocab_(tgt_vocab, ["<pad>"])
        print(f"added {num_added} tokens to target vocab")
        save_json(tgt_vocab, temp_dir / "target_vocab.json")

    # ------------- load tokenizer -------------
    tokenizer = MarianTokenizer.from_pretrained(
        temp_dir, separate_vocabs=is_vocab_separate
    )
    assert tokenizer.separate_vocabs == is_vocab_separate
    shutil.rmtree(temp_dir)

    return tokenizer


def convert(
    npz_model_path,
    yml_decoder_path,
    spm_model_path,
    vocab_path,
    tgt_vocab_path=None,
    tgt_spm_model_path=None,
):

    tokenizer = convert_tokenizer(
        vocab_path=vocab_path,
        tgt_vocab_path=tgt_vocab_path,
        spm_model_path=spm_model_path,
        tgt_spm_model_path=tgt_spm_model_path,
    )
    opus_state = OpusState(
        npz_model_path=npz_model_path,
        yml_decoder_path=yml_decoder_path,
        tokenizer=tokenizer,
    )
    model = opus_state.load_marian_model().half()
    marian_original_config = opus_state.cfg
    return model, tokenizer, marian_original_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz-model-path", type=str, help="Path to marian model file")
    parser.add_argument(
        "--yml-decoder-path", type=str, help="Path to marian decoder config"
    )
    parser.add_argument("--spm-model-path", type=str, help="Path to marian model file")
    parser.add_argument(
        "--tgt-spm-model-path",
        type=str,
        default=None,
        help="Path to marian target vocab path if vocab is not shared",
    )
    parser.add_argument("--vocab-path", type=str, help="Path to marian vocab path")
    parser.add_argument(
        "--tgt-vocab-path",
        type=str,
        default=None,
        help="Path to marian target vocab path if vocab is not shared",
    )
    parser.add_argument(
        "--dest-dir", type=str, help="Path to the output PyTorch model."
    )
    args = parser.parse_args()

    model, tokenizer, marian_original_config = convert(
        npz_model_path=args.npz_model_path,
        yml_decoder_path=args.yml_decoder_path,
        spm_model_path=args.spm_model_path,
        vocab_path=args.vocab_path,
        tgt_vocab_path=args.tgt_vocab_path,
        tgt_spm_model_path=args.tgt_spm_model_path,
    )

    Path(args.dest_dir).mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(args.dest_dir)
    tokenizer.from_pretrained(args.dest_dir)  # sanity check
    model.save_pretrained(args.dest_dir)
    model.from_pretrained(args.dest_dir)  # sanity check
    save_json(marian_original_config, f"{args.dest_dir}/marian_original_config.json")
    print(f"Saved HF model and tokenizer to {args.dest_dir}")
