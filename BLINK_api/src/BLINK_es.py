import argparse
import pathlib
import json
import sys
import os

from tqdm import tqdm
import logging
import torch
import numpy as np

from hydra.utils import to_absolute_path
from hydra import compose, initialize
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

import blink.ner as NER
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from blink.biencoder.biencoder import BiEncoderRanker, load_biencoder
from blink.crossencoder.crossencoder import CrossEncoderRanker, load_crossencoder
from blink.biencoder.data_process import (
    process_mention_data,
    get_candidate_representation,
)
import blink.candidate_ranking.utils as utils
from blink.crossencoder.train_cross import modify, evaluate
from blink.crossencoder.data_process import prepare_crossencoder_data

# model_dir = "models" # the path where you stored the BLINK models

# config = {}

# absolute_path = pathlib.Path(__file__).parent.resolve()

# args = argparse.Namespace(**config)

# args.biencoder_model = os.path.join(absolute_path, model_dir, "biencoder_wiki_large.bin")
# args.biencoder_config = os.path.join(absolute_path, model_dir, "biencoder_wiki_large.json")
# args.crossencoder_model = os.path.join(absolute_path, model_dir, "crossencoder_wiki_large.bin")
# args.crossencoder_config = os.path.join(absolute_path, model_dir, "crossencoder_wiki_large.json")
# args.output_path = "logs/"

# args.entity_catalogue = os.path.join(absolute_path, model_dir, "entity.jsonl")
# args.entity_encoding = os.path.join(absolute_path, model_dir, "all_entities_large.t7")
# args.entities_to_add = os.path.join(absolute_path, model_dir,"entities_to_add.jsonl")
# args.new_entity_catalogue = os.path.join(absolute_path, model_dir, "entity.jsonl")
# args.index_path = os.path.join(absolute_path, model_dir, "faiss_index.pkl")

# args.fast = False

# args.test_entities = None
# args.test_mentions = None
# args.interactive = False
# args.top_k = 10
# args.faiss_index = "hnsw"

def config_to_abs_paths(config, *parameter_names):
    absolute_path = pathlib.Path(__file__).parent.resolve()
    absolute_path = pathlib.Path(absolute_path).parent.resolve()
    for param_name in parameter_names:
        param = getattr(config, param_name)
        if param is not None:
            setattr(config, param_name, os.path.join(absolute_path,param))


initialize(config_path="../configs", job_name="blink")
cfg = compose(config_name="blink")
print(OmegaConf.to_yaml(cfg))

config_to_abs_paths(cfg.model, 'biencoder_model')
config_to_abs_paths(cfg.model, 'biencoder_config')
config_to_abs_paths(cfg.model, 'crossencoder_model')
config_to_abs_paths(cfg.model, 'crossencoder_config')
config_to_abs_paths(cfg.logging, 'output_path')

args = cfg

print(args)

def load_models(args, logger=None):
    # load biencoder model
    if logger:
        logger.info("loading biencoder model")
    with open(args.model.biencoder_config) as json_file:
        biencoder_params = json.load(json_file)
        biencoder_params["path_to_model"] = args.model.biencoder_model
    biencoder = load_biencoder(biencoder_params)

    crossencoder = None
    crossencoder_params = None
    if not args.fast:
        # load crossencoder model
        if logger:
            logger.info("loading crossencoder model")
        with open(args.model.crossencoder_config) as json_file:
            crossencoder_params = json.load(json_file)
            crossencoder_params["path_to_model"] = args.model.crossencoder_model
        crossencoder = load_crossencoder(crossencoder_params)

    return (
        biencoder,
        biencoder_params,
        crossencoder,
        crossencoder_params
    )

if __name__ == '__main__':

    biencoder,biencoder_params,crossencoder,crossencoder_params = load_models(args)
