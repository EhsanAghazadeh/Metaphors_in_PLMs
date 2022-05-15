from tqdm.auto import tqdm
import pandas as pd
from IPython.display import display
import torch
import numpy as np
import shutil
import os
import json
import gc
import datetime
import torch.nn as nn
from abc import ABC, abstractmethod
import torch.optim as optim
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from scipy.special import softmax
from sklearn.metrics import f1_score
import sklearn
import psutil  # RAM usage
import re
import random
import pickle
import datasets
import copy
import sys

print(torch.__version__)

"""# Configs"""

class Dataset_info:
    def __init__(self, dataset_name, num_of_spans, max_span_length=5, ignore_classes=[], manual_text=None):
        self.dataset_name = dataset_name
        self.num_of_spans = num_of_spans
        self.ignore_classes = ignore_classes  # ignore other class in rel (semeval)
        self.manual_text = manual_text

POOL_METHOD = "attn"  # 'max', 'attn'
BATCH_SIZE = 32
#SEED = 0
LEARNING_RATE = 5e-5
DEVICE = 'cuda' if torch.cuda.is_available() else "cpu"

CACHE_LEN = 500
GPU_CACHE_LEN = 200  # 600
RAM_CACHE_LEN = 400  # 2000

print(DEVICE)

dataset_info_dict = {
    "const": Dataset_info("const", num_of_spans=1),
    "ud": Dataset_info("ud", num_of_spans=2),
    "ner": Dataset_info("ner", num_of_spans=1),
    "srl": Dataset_info("srl", num_of_spans=2),
    "coref": Dataset_info("coref", num_of_spans=2),
    "semeval": Dataset_info("semeval", num_of_spans=2),
    "dpr": Dataset_info("dpr", num_of_spans=2),
    "vua_verb": Dataset_info("vua_verb", num_of_spans=1),
    "vua_pos": Dataset_info("vua_pos", num_of_spans=1),
    "trofi": Dataset_info("trofi", num_of_spans=1),
    "lcc": Dataset_info("lcc", num_of_spans=1),
    "lcc_fa": Dataset_info("lcc_fa", num_of_spans=1),
    "lcc_es": Dataset_info("lcc_es", num_of_spans=1),
    "lcc_ru": Dataset_info("lcc_ru", num_of_spans=1),
    "lcc_en_fa": Dataset_info("lcc_en_fa", num_of_spans=1),
    "lcc_en_es": Dataset_info("lcc_en_es", num_of_spans=1),
    "lcc_en_ru": Dataset_info("lcc_en_ru", num_of_spans=1),
    "lcc_es_fa": Dataset_info("lcc_es_fa", num_of_spans=1),
    "lcc_es_ru": Dataset_info("lcc_es_ru", num_of_spans=1),
    "lcc_fa_ru": Dataset_info("lcc_fa_ru", num_of_spans=1),
    "lcc_fa_en": Dataset_info("lcc_fa_en", num_of_spans=1),
    "lcc_fa_es": Dataset_info("lcc_fa_es", num_of_spans=1),
    "lcc_ru_es": Dataset_info("lcc_ru_es", num_of_spans=1),
    "lcc_ru_fa": Dataset_info("lcc_ru_fa", num_of_spans=1),
    "lcc_ru_en": Dataset_info("lcc_ru_en", num_of_spans=1),
    "lcc_es_en": Dataset_info("lcc_es_en", num_of_spans=1),
    "cross_trofi_vua_verb": Dataset_info("cross_trofi_vua_verb", num_of_spans=1),
}

model_checkpoint = sys.argv[1]
my_dataset_info = dataset_info_dict[sys.argv[2]]
SEED = int(sys.argv[3])

SEQ2SEQ_MODEL = "t5" in model_checkpoint or "pegasus" in model_checkpoint or "bart" in model_checkpoint
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
set_seed(SEED)

"""# Prepare Dataset & Spans"""

from transformers import AutoTokenizer, AutoModel

if "glove" in model_checkpoint:
    model = AutoModel.from_pretrained("bert-base-uncased")
else:
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, add_prefix_space=True)
    tokenizer.padding_side = 'right'
    model = AutoModel.from_pretrained(model_checkpoint)

class Utils:
    def one_hot(idx, length):
        import numpy as np
        o = np.zeros(length, dtype=np.int8)
        o[idx] = 1
        return o

class Dataset_handler:
    def __init__(self, dataset_info: Dataset_info):
        self.dataset = datasets.DatasetDict()
        self.tokenized_dataset = None
        self.dataset_info = dataset_info
        self.labels_list = None
        # CACHE
        self.global_cache_counter = 0
        self.cache_last_hashable_input = ""

        if dataset_info.dataset_name == "dpr":
            self.json_to_dataset('./edge-probing-datasets/data/dpr_data/train.json', data_type="train")
            self.json_to_dataset('./edge-probing-datasets/data/dpr_data/dev.json', data_type="dev")
            self.json_to_dataset('./edge-probing-datasets/data/dpr_data/test.json', data_type="test")
        elif dataset_info.dataset_name == "const":
            frac = 0.01
            self.json_to_dataset('./ontonotes_data/const/train.json', data_type="train", fraction = frac, sample_from_head=True)
            self.json_to_dataset('./ontonotes_data/const/conll-2012-test.json', data_type="dev", fraction = frac, sample_from_head=True)
            self.json_to_dataset('./ontonotes_data/const/test.json', data_type="test", fraction = frac, sample_from_head=True)
        elif dataset_info.dataset_name == "ud":
            frac = 1
            self.json_to_dataset('./edge-probing-datasets/data/ud_data/en_ewt-ud-train.json', data_type="train", fraction = frac)
            self.json_to_dataset('./edge-probing-datasets/data/ud_data/en_ewt-ud-dev.json', data_type="dev", fraction = frac)
            self.json_to_dataset('./edge-probing-datasets/data/ud_data/en_ewt-ud-test.json', data_type="test", fraction = frac)
        elif dataset_info.dataset_name == "semeval":
            frac = 1
            self.json_to_dataset('./edge-probing-datasets/data/semeval_data/train.0.85.json', data_type="train", fraction = frac, ignore_classes = self.dataset_info.ignore_classes)
            self.json_to_dataset('./edge-probing-datasets/data/semeval_data/test.json', data_type="dev", fraction = 0.01, ignore_classes = self.dataset_info.ignore_classes)
            self.json_to_dataset('./edge-probing-datasets/data/semeval_data/test.json', data_type="test", fraction = frac, ignore_classes = self.dataset_info.ignore_classes)
        elif dataset_info.dataset_name == "srl":
            frac = 1
            self.json_to_dataset('./ontonotes_data/srl/train.json', data_type="train", fraction = frac)
            self.json_to_dataset('./ontonotes_data/srl/conll-2012-test.json', data_type="dev", fraction = frac)
            self.json_to_dataset('./ontonotes_data/srl/test.json', data_type="test", fraction = frac)
        elif dataset_info.dataset_name == "ner":
            frac = 1
            self.json_to_dataset('./ontonotes_data/ner/train.json', data_type="train", fraction = frac)
            self.json_to_dataset('./ontonotes_data/ner/conll-2012-test.json', data_type="dev", fraction = frac)
            self.json_to_dataset('./ontonotes_data/ner/test.json', data_type="test", fraction = frac)
        elif dataset_info.dataset_name == "coref":
            frac = 1
            self.json_to_dataset('./ontonotes_data/coref/train.json', data_type="train", fraction = frac)
            self.json_to_dataset('./ontonotes_data/coref/development.json', data_type="dev", fraction = frac)
            self.json_to_dataset('./ontonotes_data/coref/test.json', data_type="test", fraction = frac)
        elif dataset_info.dataset_name == "offenseval2019":
            frac = 1
            self.json_to_dataset('./edge-probing-datasets/toxicity/offenseval2019/train.json', data_type="train", fraction = frac)
            self.json_to_dataset('./edge-probing-datasets/toxicity/offenseval2019/dev.json', data_type="dev", fraction = frac)
            self.json_to_dataset('./edge-probing-datasets/toxicity/offenseval2019/test.json', data_type="test", fraction = frac)
        elif dataset_info.dataset_name == "hatexplain":
            frac = 1
            self.json_to_dataset('./edge-probing-datasets/toxicity/hatexplain/train.json', data_type="train", fraction = frac)
            self.json_to_dataset('./edge-probing-datasets/toxicity/hatexplain/dev.json', data_type="dev", fraction = frac)
            self.json_to_dataset('./edge-probing-datasets/toxicity/hatexplain/test.json', data_type="test", fraction = frac)
        elif dataset_info.dataset_name == "hatexplain-fullspan":
            frac = 1
            self.json_to_dataset('./edge-probing-datasets/toxicity/hatexplain/train.json', data_type="train", fraction = frac, to_sentence_span=True)
            self.json_to_dataset('./edge-probing-datasets/toxicity/hatexplain/dev.json', data_type="dev", fraction = frac, to_sentence_span=True)
            self.json_to_dataset('./edge-probing-datasets/toxicity/hatexplain/test.json', data_type="test", fraction = frac, to_sentence_span=True)
        elif dataset_info.dataset_name == "jigsaw_bias":
            train_frac = 0.1
            test_frac = 1
            self.json_to_dataset('./edge-probing-datasets/toxicity/jigsaw_bias/train100.json', data_type="train", fraction = train_frac, keep_order=False)
            self.json_to_dataset('./edge-probing-datasets/toxicity/jigsaw_bias/test100.json', data_type="dev", fraction = 0.001)
            self.json_to_dataset('./edge-probing-datasets/toxicity/jigsaw_bias/test100.json', data_type="test", fraction = test_frac)
        elif dataset_info.dataset_name == "vua_verb":
            frac = 1
            self.json_to_dataset('./edge-probing-datasets/metaphor/vua/verb_train.json', data_type="train", fraction = frac)
            self.json_to_dataset('./edge-probing-datasets/metaphor/vua/verb_test.json', data_type="dev", fraction = 0.01)
            self.json_to_dataset('./edge-probing-datasets/metaphor/vua/verb_test.json', data_type="test", fraction = frac)
        elif dataset_info.dataset_name == "vua_pos":
            frac = 1
            self.json_to_dataset('./edge-probing-datasets/metaphor/vua/pos_train.json', data_type="train", fraction = frac)
            self.json_to_dataset('./edge-probing-datasets/metaphor/vua/pos_test.json', data_type="dev", fraction = 0.01)
            self.json_to_dataset('./edge-probing-datasets/metaphor/vua/pos_test.json', data_type="test", fraction = frac)
        elif dataset_info.dataset_name == "trofi":
            frac = 1
            self.json_to_dataset('./edge-probing-datasets/metaphor/trofi/train.json', data_type="train", fraction = frac, keep_order=False)
            self.json_to_dataset('./edge-probing-datasets/metaphor/trofi/test.json', data_type="dev", fraction = 0.01)
            self.json_to_dataset('./edge-probing-datasets/metaphor/trofi/test.json', data_type="test", fraction = frac)
        elif dataset_info.dataset_name == "trofi_nospan":
            frac = 1
            self.json_to_dataset('./edge-probing-datasets/metaphor/trofi/train.json', data_type="train", fraction = frac, keep_order=False, to_sentence_span=True)
            self.json_to_dataset('./edge-probing-datasets/metaphor/trofi/test.json', data_type="dev", fraction = 0.01, to_sentence_span=True)
            self.json_to_dataset('./edge-probing-datasets/metaphor/trofi/test.json', data_type="test", fraction = frac, to_sentence_span=True)
        elif dataset_info.dataset_name == "lcc":
            frac = 1
            self.json_to_dataset('./edge-probing-datasets/metaphor/lcc/en/en_train10_current.json', data_type="train", fraction = 0.43984008)
            self.json_to_dataset('./edge-probing-datasets/metaphor/lcc/en/en_test10_current.json', data_type="dev", fraction = 0.01)
            self.json_to_dataset('./edge-probing-datasets/metaphor/lcc/en/en_test10_current.json', data_type="test", fraction = frac)
        elif dataset_info.dataset_name == "lcc_src_concept":
            frac = 1
            self.json_to_dataset('./edge-probing-datasets/metaphor/lcc/en/en_train10_src_concept_current.json', data_type="train", fraction = frac, ignore_classes = self.dataset_info.ignore_classes)
            self.json_to_dataset('./edge-probing-datasets/metaphor/lcc/en/en_test10_src_concept_current.json', data_type="dev", fraction = 0.01, ignore_classes = self.dataset_info.ignore_classes)
            self.json_to_dataset('./edge-probing-datasets/metaphor/lcc/en/en_test10_src_concept_current.json', data_type="test", fraction = frac, ignore_classes = self.dataset_info.ignore_classes)
        elif dataset_info.dataset_name == "lcc_src_target_concept":
            frac = 1
            self.json_to_dataset('./edge-probing-datasets/metaphor/lcc/en/en_train10_src_concept_current.json', data_type="train", fraction = frac, ignore_classes = self.dataset_info.ignore_classes)
            self.json_to_dataset('./edge-probing-datasets/metaphor/lcc/en/en_test10_src_concept_current.json', data_type="dev", fraction = 0.01, ignore_classes = self.dataset_info.ignore_classes)
            self.json_to_dataset('./edge-probing-datasets/metaphor/lcc/en/en_test10_src_concept_current.json', data_type="test", fraction = frac, ignore_classes = self.dataset_info.ignore_classes)
        elif dataset_info.dataset_name == "lcc_fa":
            frac = 1
            self.json_to_dataset('./edge-probing-datasets/metaphor/lcc/fa/fa_train10_current.json', data_type="train", fraction = 0.94571799)
            self.json_to_dataset('./edge-probing-datasets/metaphor/lcc/fa/fa_test10_current.json', data_type="dev", fraction = 0.01)
            self.json_to_dataset('./edge-probing-datasets/metaphor/lcc/fa/fa_test10_current.json', data_type="test", fraction = frac)
        elif dataset_info.dataset_name == "lcc_es":
            frac = 1
            self.json_to_dataset('./edge-probing-datasets/metaphor/lcc/es/es_train10_current.json', data_type="train", fraction = 0.60313082)
            self.json_to_dataset('./edge-probing-datasets/metaphor/lcc/es/es_test10_current.json', data_type="dev", fraction = 0.01)
            self.json_to_dataset('./edge-probing-datasets/metaphor/lcc/es/es_test10_current.json', data_type="test", fraction = frac)
        elif dataset_info.dataset_name == "lcc_ru":
            frac = 1
            self.json_to_dataset('./edge-probing-datasets/metaphor/lcc/ru/ru_train10_current.json', data_type="train", fraction = frac)
            self.json_to_dataset('./edge-probing-datasets/metaphor/lcc/ru/ru_test10_current.json', data_type="dev", fraction = 0.01)
            self.json_to_dataset('./edge-probing-datasets/metaphor/lcc/ru/ru_test10_current.json', data_type="test", fraction = frac)
        elif dataset_info.dataset_name == "lcc_en_fa":
            frac = 1
            self.json_to_dataset('./edge-probing-datasets/metaphor/lcc/en/en_train10_current.json', data_type="train", fraction = 0.43984008)
            # self.json_to_dataset('./edge-probing-datasets/metaphor/lcc/en/en_train10_current.json', data_type="train", fraction = 1)
            self.json_to_dataset('./edge-probing-datasets/metaphor/lcc/en/en_test10_current.json', data_type="dev", fraction = 0.01)
            self.json_to_dataset('./edge-probing-datasets/metaphor/lcc/fa/fa_test10_current.json', data_type="test", fraction = frac)
        elif dataset_info.dataset_name == "lcc_en_es":
            frac = 1
            self.json_to_dataset('./edge-probing-datasets/metaphor/lcc/en/en_train10_current.json', data_type="train", fraction = 0.43984008)
            self.json_to_dataset('./edge-probing-datasets/metaphor/lcc/en/en_test10_current.json', data_type="dev", fraction = 0.01)
            self.json_to_dataset('./edge-probing-datasets/metaphor/lcc/es/es_test10_current.json', data_type="test", fraction = frac)
        elif dataset_info.dataset_name == "lcc_en_ru":
            frac = 1
            self.json_to_dataset('./edge-probing-datasets/metaphor/lcc/en/en_train10_current.json', data_type="train", fraction = 0.43984008)
            self.json_to_dataset('./edge-probing-datasets/metaphor/lcc/en/en_test10_current.json', data_type="dev", fraction = 0.01)
            self.json_to_dataset('./edge-probing-datasets/metaphor/lcc/ru/ru_test10_current.json', data_type="test", fraction = frac)
        elif dataset_info.dataset_name == "lcc_es_fa":
            frac = 1
            self.json_to_dataset('./edge-probing-datasets/metaphor/lcc/es/es_train10_current.json', data_type="train", fraction = 0.60313082)
            self.json_to_dataset('./edge-probing-datasets/metaphor/lcc/es/es_test10_current.json', data_type="dev", fraction = 0.01)
            self.json_to_dataset('./edge-probing-datasets/metaphor/lcc/fa/fa_test10_current.json', data_type="test", fraction = frac)
        elif dataset_info.dataset_name == "lcc_es_ru":
            frac = 1
            self.json_to_dataset('./edge-probing-datasets/metaphor/lcc/es/es_train10_current.json', data_type="train", fraction = 0.60313082)
            self.json_to_dataset('./edge-probing-datasets/metaphor/lcc/es/es_test10_current.json', data_type="dev", fraction = 0.01)
            self.json_to_dataset('./edge-probing-datasets/metaphor/lcc/ru/ru_test10_current.json', data_type="test", fraction = frac)
        elif dataset_info.dataset_name == "lcc_es_en":
            frac = 1
            self.json_to_dataset('./edge-probing-datasets/metaphor/lcc/es/es_train10_current.json', data_type="train", fraction = 0.60313082)
            self.json_to_dataset('./edge-probing-datasets/metaphor/lcc/es/es_test10_current.json', data_type="dev", fraction = 0.01)
            self.json_to_dataset('./edge-probing-datasets/metaphor/lcc/en/en_test10_current.json', data_type="test", fraction = frac)
        elif dataset_info.dataset_name == "lcc_fa_ru":
            frac = 1
            self.json_to_dataset('./edge-probing-datasets/metaphor/lcc/fa/fa_train10_current.json', data_type="train", fraction = 0.94571799)
            self.json_to_dataset('./edge-probing-datasets/metaphor/lcc/fa/fa_test10_current.json', data_type="dev", fraction = 0.01)
            self.json_to_dataset('./edge-probing-datasets/metaphor/lcc/ru/ru_test10_current.json', data_type="test", fraction = frac)
        elif dataset_info.dataset_name == "lcc_fa_en":
            frac = 1
            self.json_to_dataset('./edge-probing-datasets/metaphor/lcc/fa/fa_train10_current.json', data_type="train", fraction = 0.94571799)
            self.json_to_dataset('./edge-probing-datasets/metaphor/lcc/fa/fa_test10_current.json', data_type="dev", fraction = 0.01)
            self.json_to_dataset('./edge-probing-datasets/metaphor/lcc/en/en_test10_current.json', data_type="test", fraction = frac)
        elif dataset_info.dataset_name == "lcc_fa_es":
            frac = 1
            self.json_to_dataset('./edge-probing-datasets/metaphor/lcc/fa/fa_train10_current.json', data_type="train", fraction = 0.94571799)
            self.json_to_dataset('./edge-probing-datasets/metaphor/lcc/fa/fa_test10_current.json', data_type="dev", fraction = 0.01)
            self.json_to_dataset('./edge-probing-datasets/metaphor/lcc/es/es_test10_current.json', data_type="test", fraction = frac)
        elif dataset_info.dataset_name == "lcc_ru_en":
            frac = 1
            self.json_to_dataset('./edge-probing-datasets/metaphor/lcc/ru/ru_train10_current.json', data_type="train", fraction = frac)
            self.json_to_dataset('./edge-probing-datasets/metaphor/lcc/ru/ru_test10_current.json', data_type="dev", fraction = 0.01)
            self.json_to_dataset('./edge-probing-datasets/metaphor/lcc/en/en_test10_current.json', data_type="test", fraction = frac)
        elif dataset_info.dataset_name == "lcc_ru_es":
            frac = 1
            self.json_to_dataset('./edge-probing-datasets/metaphor/lcc/ru/ru_train10_current.json', data_type="train", fraction = frac)
            self.json_to_dataset('./edge-probing-datasets/metaphor/lcc/ru/ru_test10_current.json', data_type="dev", fraction = 0.01)
            self.json_to_dataset('./edge-probing-datasets/metaphor/lcc/es/es_test10_current.json', data_type="test", fraction = frac)
        elif dataset_info.dataset_name == "lcc_ru_fa":
            frac = 1
            self.json_to_dataset('./edge-probing-datasets/metaphor/lcc/ru/ru_train10_current.json', data_type="train", fraction = frac)
            self.json_to_dataset('./edge-probing-datasets/metaphor/lcc/ru/ru_test10_current.json', data_type="dev", fraction = 0.01)
            self.json_to_dataset('./edge-probing-datasets/metaphor/lcc/fa/fa_test10_current.json', data_type="test", fraction = frac)
        elif dataset_info.dataset_name == "cross_trofi_vua_verb":
            frac = 1
            self.json_to_dataset('./edge-probing-datasets/metaphor/trofi/train.json', data_type="train", fraction = frac)
            self.json_to_dataset('./edge-probing-datasets/metaphor/trofi/train.json', data_type="dev", fraction = 0.01)
            self.json_to_dataset('./edge-probing-datasets/metaphor/vua/verb_test.json', data_type="test", fraction = frac)
        elif dataset_info.dataset_name == "lcc_en+fa_fa":
            frac = 1
            self.merge_files(['./edge-probing-datasets/metaphor/lcc/fa/fa_train10_current.json', 
                              './edge-probing-datasets/metaphor/lcc/en/en_train10_current.json'],
                             "merged.json")
            self.json_to_dataset('merged.json', data_type="train", keep_order=False, fraction = frac)
            self.json_to_dataset('merged.json', data_type="dev", fraction = 0.01)
            self.json_to_dataset('./edge-probing-datasets/metaphor/lcc/fa/fa_test10_current.json', data_type="test", fraction = frac)
        elif dataset_info.dataset_name == "lcc_en+es_es":
            frac = 1
            self.merge_files(['./edge-probing-datasets/metaphor/lcc/es/es_train10_current.json', 
                              './edge-probing-datasets/metaphor/lcc/en/en_train10_current.json'],
                             "merged.json")
            self.json_to_dataset('merged.json', data_type="train", keep_order=False, fraction = frac)
            self.json_to_dataset('merged.json', data_type="dev", fraction = 0.01)
            self.json_to_dataset('./edge-probing-datasets/metaphor/lcc/es/es_test10_current.json', data_type="test", fraction = frac)
        elif dataset_info.dataset_name == "lcc_en+ru_ru":
            frac = 1
            self.merge_files(['./edge-probing-datasets/metaphor/lcc/ru/ru_train10_current.json', 
                              './edge-probing-datasets/metaphor/lcc/en/en_train10_current.json'],
                             "merged.json")
            self.json_to_dataset('merged.json', data_type="train", keep_order=False, fraction = frac)
            self.json_to_dataset('merged.json', data_type="dev", fraction = 0.01)
            self.json_to_dataset('./edge-probing-datasets/metaphor/lcc/ru/ru_test10_current.json', data_type="test", fraction = frac)
        elif dataset_info.dataset_name == "manual":
            frac = 1
            f = open("./manual_dataset.json", "w")
            f.write('{"text": "' + dataset_info.manual_text + '", "targets": [{"span1": [0, 0], "label": "' + my_dataset_handler.labels_list[0] + '"}]}')
            f.write('\n{"text": "' + dataset_info.manual_text + '", "targets": [{"span1": [0, 0], "label": "' + my_dataset_handler.labels_list[1] + '"}]}')
            f.close()
            self.json_to_dataset('./manual_dataset.json', data_type="train", fraction = frac, to_sentence_span=True)
            self.json_to_dataset('./manual_dataset.json', data_type="dev", fraction = frac, to_sentence_span=True)
            self.json_to_dataset('./manual_dataset.json', data_type="test", fraction = frac, to_sentence_span=True)
        else:
            throw("Error: Unkown dataset name!")

        print("⌛ Tokenizing Dataset and Adding One Hot Representation of Labels")
        self.tokenized_dataset = self.tokenize_input_and_one_hot_labels(self.dataset)
        # self.tokenized_dataset = self.tokenize_dataset(self.dataset)
        # print("⌛ Adding One Hot Representation of Labels")
        # self.tokenized_dataset = self.one_hot_dataset_labels(self.tokenized_dataset)
        

    # Private:
    def merge_files(self, file_addresses: [], output_address: str):
        data = ""
        for file_address in file_addresses:
            with open(file_address) as fp:
                data += fp.read()
        with open (output_address, 'w') as fp:
            fp.write(data)

    def json_to_dataset(self, json_path, data_type="train", fraction=1, ignore_classes=[], keep_order=False, sample_from_head=False, to_sentence_span=False):
        data_df = self.json_to_df(json_path, to_sentence_span)
        data_df = data_df[~data_df["label"].isin(ignore_classes)]
        if sample_from_head:
            data_df = data_df.head(int(len(data_df) * fraction))
        else:
            if keep_order:
                data_df = data_df.sample(frac=fraction, random_state=SEED).sort_index().reset_index(drop=True)
            else:
                data_df = data_df.sample(frac=fraction, random_state=SEED).reset_index(drop=True)
        self.dataset[data_type] = datasets.Dataset.from_pandas(data_df)
        return self.dataset
    
    def tokenize_input_and_one_hot_labels(self, dataset):
        train_df = pd.DataFrame(dataset["train"]["label"], columns=['label'])
        dev_df = pd.DataFrame(dataset["dev"]["label"], columns=['label'])
        test_df = pd.DataFrame(dataset["test"]["label"], columns=['label'])
        self.labels_list = list(set(train_df["label"].unique()).union
                               (set(dev_df["label"].unique())).union
                               (set(test_df["label"].unique())))
        self.label_to_index = dict()
        for idx, l in enumerate(self.labels_list):
            self.label_to_index[l] = idx
        
        if "glove" in model_checkpoint or "elmo" in model_checkpoint:
            tokenized_one_hot_dataset = dataset.map(tokenize_and_one_hot_glove,
                                                    fn_kwargs={"label_to_index": self.label_to_index,
                                                            "labels_len": len(self.label_to_index),
                                                            "one_hot_func": Utils.one_hot,
                                                            "num_of_spans": self.dataset_info.num_of_spans
                                                            },
                                                    batched=False,
                                                    num_proc=None)
        else:
            tokenized_one_hot_dataset = dataset.map(tokenize_and_one_hot,
                                                    fn_kwargs={"label_to_index": self.label_to_index,
                                                            "labels_len": len(self.label_to_index),
                                                            "tokenizer": tokenizer,
                                                            "one_hot_func": Utils.one_hot,
                                                            "num_of_spans": self.dataset_info.num_of_spans
                                                            },
                                                    batched=False,
                                                    num_proc=None)
        return tokenized_one_hot_dataset

    # Preprocesses
    def lcc_preprocess(self, target, instance):
        if "lcc" in self.dataset_info.dataset_name and "src" not in self.dataset_info.dataset_name:
            target["label"] = float(target["label"])
            if 0.0 <= target["label"] < 0.5:
                target["label"] = "Non-metaphor"
            elif 1.5 < target["label"] <= 3.0:
                target["label"] = "Metaphor"
            else:
                return None, None
        return target, instance
    
    def lcc_src_concept_preprocess(self, target, instance):
        if self.dataset_info.dataset_name == "lcc_src_concept":
            target["span1"] = target["span2"]
            score = float(target["score"])
            if score >= 2:
                return target, instance
            else:
                return None, None
        return target, instance

    def hatexplain_preprocess(self, target, instance):
        # if self.dataset_info.dataset_name == "hatexplain":
            # span_len = target["span1"][1] - target["span1"][0]
            # if target["label"] == "Normal":
            #     hatexplain_distribution_normal[span_len] += 1
            # else:
            #     hatexplain_distribution_toxic[span_len] += 1
            
            # instance["text"] = re.sub(r'[^A-Za-z0-9 ]+', '', instance["text"])  # Alphanumeric + Space
            # if True or target["label"] == "Normal":
            #     e = len(instance["text"].split())
            #     target["span1"] = [0, e]
        return target, instance

    def hatexplain_fullspan_preprocess(self, target, instance):
        if self.dataset_info.dataset_name == "hatexplain-fullspan":
            instance["text"] = re.sub(r'[^A-Za-z0-9 ]+', '', instance["text"])  # Alphanumeric + Space
        return target, instance
    
    def lcc_src_target_concept_preprocess(self, target, instance):
        if self.dataset_info.dataset_name == "lcc_src_target_concept":
            target["label"] = "(" + target["label"] + "," + instance["targetConcept"] + ")"
        return target, instance

    def json_to_df(self, json_path, to_sentence_span=False):
        pre_processes = [self.lcc_preprocess, self.lcc_src_concept_preprocess, 
                         self.hatexplain_fullspan_preprocess, 
                         self.lcc_src_target_concept_preprocess,
                         self.hatexplain_preprocess]
        with open(json_path, encoding='utf-8') as file:
            c = 0
            data_list = list()
            for line in file:
                # print(c, end=",")
                c += 1
                instance = json.loads(line)

                if self.cache_last_hashable_input != repr(instance["text"]):
                    self.cache_last_hashable_input = repr(instance["text"])
                    self.global_cache_counter += 1

                for target in instance["targets"]:
                    for pre_process in pre_processes:
                        target, instance = pre_process(target, instance)
                    if target == None:
                        break
                    if self.dataset_info.num_of_spans == 2:
                        data_list.append({"text": instance["text"],
                                        "span1": target["span1"],
                                        "span2": target["span2"],
                                        "label": str(target["label"]),
                                        "cache_id": self.global_cache_counter})
                    elif self.dataset_info.num_of_spans == 1:
                        if to_sentence_span:
                            target["span1"][0] = 0
                            target["span1"][-1] = len(instance["text"].split())
                        data_list.append({"text": instance["text"],
                                        "span1": target["span1"],
                                        "label": str(target["label"]),
                                        "cache_id": self.global_cache_counter})
        return pd.DataFrame.from_dict(data_list)

def tokenize_and_one_hot_glove(examples, **fn_kwargs):
    # tokenize and align spans
    one_hot_func = fn_kwargs["one_hot_func"]
    num_of_spans = fn_kwargs["num_of_spans"]
    tokenized_inputs = {"text": examples["text"].lower().split()}

    tokenized_inputs["span1"] = [examples["span1"][0], examples["span1"][1]]
    tokenized_inputs["span1_len"] = tokenized_inputs["span1"][1] - tokenized_inputs["span1"][0]
    if num_of_spans == 2:
        tokenized_inputs["span2"] = [examples["span2"][0], examples["span2"][1]]
        tokenized_inputs["span2_len"] = tokenized_inputs["span2"][1] - tokenized_inputs["span2"][0]
    # One hot
    label_to_index = fn_kwargs["label_to_index"]
    labels_len = fn_kwargs["labels_len"]
    tokenized_inputs["one_hot_label"] = one_hot_func(label_to_index[examples["label"]], labels_len)
    return tokenized_inputs

cached_tokenized_input = {}
cached_onehot = {}
def tokenize_and_one_hot(examples, **fn_kwargs):
    # tokenize and align spans
    thread_tokenizer = fn_kwargs["tokenizer"]
    one_hot_func = fn_kwargs["one_hot_func"]
    num_of_spans = fn_kwargs["num_of_spans"]
    
    if repr(examples["text"]) in cached_tokenized_input:
        tokenized_inputs = cached_tokenized_input[repr(examples["text"])]
    else:
        tokenized_inputs = thread_tokenizer(examples["text"].split(), is_split_into_words=True)  # Must be splitted for tokenizer to word_ids works fine. (test e-mail!)
        # cached_tokenized_input = {}  # Free For RAM (Just Last One Cached)
        cached_tokenized_input[repr(examples["text"])] = tokenized_inputs
    # tokenized_inputs = tokenizer(examples["text"], truncation=True, is_split_into_words=True, padding="max_length", max_length=210)
    def align_span(word_ids, start_word_id, end_word_id):
        span = [0, 0]
        if start_word_id not in word_ids:
            print("Warning: There is no", start_word_id, "in", word_ids, examples["text"].split(), examples["label"])
            start_word_id -= 1
        span[0] = word_ids.index(start_word_id)  # First occurance
        if end_word_id - 1 not in word_ids[::-1]:
            print("Warning: There is no", end_word_id - 1, "in", word_ids, examples["text"].split(), examples["label"])
            end_word_id -= 1
        span[1] = len(word_ids) - 1 - word_ids[::-1].index(end_word_id - 1) + 1  # Last occurance (+1 for open range)
        return span

    # tokenized_inputs["span1"] = [0, 0]
    # tokenized_inputs["span1"][0] = word_ids.index(examples["span1"][0])  # First occurance
    # tokenized_inputs["span1"][1] = len(word_ids) - 1 - word_ids[::-1].index(examples["span1"][1] - 1) + 1  # Last occurance (+1 for open range)
    word_ids = tokenized_inputs.word_ids()
    tokenized_inputs["span1"] = align_span(word_ids, examples["span1"][0], examples["span1"][1])
    tokenized_inputs["span1_len"] = tokenized_inputs["span1"][1] - tokenized_inputs["span1"][0]
    if num_of_spans == 2:
        # tokenized_inputs["span2"] = [0, 0]
        # tokenized_inputs["span2"][0] = word_ids.index(examples["span2"][0])  # First occurance
        # tokenized_inputs["span2"][1] = len(word_ids) - 1 - word_ids[::-1].index(examples["span2"][1] - 1) + 1  # Last occurance
        tokenized_inputs["span2"] = align_span(word_ids, examples["span2"][0], examples["span2"][1])
        tokenized_inputs["span2_len"] = tokenized_inputs["span2"][1] - tokenized_inputs["span2"][0]
    # One hot
    label_to_index = fn_kwargs["label_to_index"]
    labels_len = fn_kwargs["labels_len"]
    if examples["label"] in cached_onehot:
        tokenized_inputs["one_hot_label"] = cached_onehot[examples["label"]]
    else:
        tokenized_inputs["one_hot_label"] = one_hot_func(label_to_index[examples["label"]], labels_len)
        cached_onehot[examples["label"]] = tokenized_inputs["one_hot_label"]
    return tokenized_inputs

my_dataset_handler = Dataset_handler(my_dataset_info);

# Check
rnd_idx = np.random.randint(1000)
# rnd_idx = 58000
part = "train"
display(pd.DataFrame(my_dataset_handler.tokenized_dataset['train'][0:3]))
display(pd.DataFrame(my_dataset_handler.tokenized_dataset['test'][0:3]))
print("idx =", rnd_idx)
print(my_dataset_handler.tokenized_dataset)
print("Original Spans:", my_dataset_handler.dataset[part][rnd_idx])
print("Tokenized Spans:", my_dataset_handler.tokenized_dataset[part][rnd_idx])
if "glove" in model_checkpoint or "elmo" in model_checkpoint:
    test_tokens = my_dataset_handler.tokenized_dataset[part][rnd_idx]["text"]
else:
    test_tokens = tokenizer.convert_ids_to_tokens(my_dataset_handler.tokenized_dataset[part][rnd_idx]["input_ids"])
print(test_tokens)

s10, s11 = my_dataset_handler.tokenized_dataset[part][rnd_idx]["span1"][0], my_dataset_handler.tokenized_dataset[part][rnd_idx]["span1"][-1]
print("span1:", s10, s11, test_tokens[s10:s11])
if my_dataset_info.num_of_spans == 2:
    s20, s21 = my_dataset_handler.tokenized_dataset[part][rnd_idx]["span2"][0], my_dataset_handler.tokenized_dataset[part][rnd_idx]["span2"][-1]
    print("span2:", s20, s21, test_tokens[s20:s21])
print("label:", my_dataset_handler.tokenized_dataset[part][rnd_idx]["label"])

stats = pd.DataFrame(my_dataset_handler.tokenized_dataset[part]["label"], columns=['label'])["label"].value_counts()
print(stats.to_string())
print(list(stats.index))
print("|Labels| =", len(stats))
stats.plot(kind='barh', color="green", figsize=(10, 9));

#  What if Google Morphed Into GoogleOS ? cc 
# span1: 11 12 ['and']
# span2: 13 14 ['e']
# label: cc
# BUG? 17

"""# Edge Probe"""

class SpanRepr(ABC, nn.Module):
    """Abstract class describing span representation."""

    def __init__(self, input_dim, use_proj=False, proj_dim=256):
        super(SpanRepr, self).__init__()
        self.input_dim = input_dim  # embedding dim or proj dim
        self.proj_dim = proj_dim
        self.use_proj = use_proj

    @abstractmethod
    def forward(self, spans, attention_mask):
        """ 
        input:
            spans: [batch_size, layers, span_max_len, proj_dim/embedding_dim] ~ [32, 13, 4, 256]
            attention_mask: [batch_size, span_max_len] ~ [32, 4]
        returns:
            [32, 13, 256]
        """
        raise NotImplementedError

    def get_input_dim(self):
        return self.input_dim

class MaxSpanRepr(SpanRepr, nn.Module):
    """Class implementing the max-pool span representation."""

    def forward(self, spans, attention_mask):
        # TODO: Vectorize this
        # for i in range(len(attention_mask)):
        #     for j in range(len(attention_mask[i])):
        #         if attention_mask[i][j] == 0:
        #             spans[i, :, j, :] = -1e10

        span_masks_shape = attention_mask.shape
        span_masks = attention_mask.reshape(
            span_masks_shape[0],
            1,
            span_masks_shape[1],
            1
        ).expand_as(spans)
        attention_spans = spans * span_masks - 1e10 * (1 - span_masks)

        max_span_repr, max_idxs = torch.max(attention_spans, dim=-2)
        # print(max_span_repr.shape)
        return max_span_repr

class AttnSpanRepr(SpanRepr, nn.Module):
    """Class implementing the attention-based span representation."""

    def __init__(self, input_dim, use_proj=False, proj_dim=256, use_endpoints=False):
        """If use_endpoints is true then concatenate the end points to attention-pooled span repr.
        Otherwise just return the attention pooled term. (use_endpoints Not Implemented)
        """
        super(AttnSpanRepr, self).__init__(input_dim, use_proj=use_proj, proj_dim=proj_dim)
        self.use_endpoints = use_endpoints
        # input_dim is embedding_dim or proj dim
        # print("input_dim", input_dim)
        self.attention_params = nn.Linear(input_dim, 1)  # Learn a weight for each token: z(k)i = W(k)att e(k)i
        self.last_attention_wts = None
        # Initialize weight to zero weight
        # self.attention_params.weight.data.fill_(0)
        # self.attention_params.bias.data.fill_(0)

    def forward(self, spans, attention_mask):
        """ 
        input:
            spans: [batch_size, layers, span_max_len, proj_dim/embedding_dim] ~ [32, 13, 4, 256]
            attention_mask: [batch_size, span_max_len] ~ [32, 4]
        returns:
            [32, 13, 256]
        """
        # if self.use_proj:
        #     encoded_input = self.proj(encoded_input)

        # span_mask = get_span_mask(start_ids, end_ids, encoded_input.shape[1])
        # attn_mask = torch.zeros(spans.shape, device=DEVICE)
        # print(datetime.datetime.now().time(), "a1")
        # print(attention_mask.shape)
        # for i in range(len(attention_mask)):
        #     for j in range(len(attention_mask[i])):
        #         if attention_mask[i][j] == 0:
        #             attn_mask[i, :, j, :] = -1e10

        span_masks_shape = attention_mask.shape
        span_masks = attention_mask.reshape(
            span_masks_shape[0],
            1,
            span_masks_shape[1],
            1
        ).expand_as(spans)
        attn_mask = - 1e10 * (1 - span_masks)
        
        # print(datetime.datetime.now().time(), "a2")

        # attn_mask = (1 - span_mask) * (-1e10)
        attn_logits = self.attention_params(spans) + attn_mask  # Decreasing the attention of padded spans by -1e10
        attention_wts = nn.functional.softmax(attn_logits, dim=-2)
        attention_term = torch.sum(attention_wts * spans, dim=-2)

        self.last_attention_wts = attention_wts   # Save for later analysis
        
        # if self.use_endpoints:
        #     batch_size = encoded_input.shape[0]
        #     h_start = encoded_input[torch.arange(batch_size), start_ids, :]
        #     h_end = encoded_input[torch.arange(batch_size), end_ids, :]
        #     return torch.cat([h_start, h_end, attention_term], dim=1)
        # else:
        #     return attention_term

        # print(spans.shape, attn_mask.shape)
        # print("attn_mask", attn_mask.shape)
        # print(attn_mask[sidx, :, :, 0:2])
        # print("attn_logits", attn_logits.shape)
        # print(attn_logits[sidx])
        # print("attention_wts", attention_wts.shape)
        # print(attention_wts[sidx, :, :, 0:2])
        # print("attention_term", attention_term.shape)
        # print(attention_term[sidx, :, 0:2])
        return attention_term.float()

def get_span_module(input_dim, method="max", use_proj=False, proj_dim=256):
    """Initializes the appropriate span representation class and returns the object.
    """
    if method == "avg":
        return AvgSpanRepr(input_dim, use_proj=use_proj, proj_dim=proj_dim)
    elif method == "max":
        return MaxSpanRepr(input_dim, use_proj=use_proj, proj_dim=proj_dim)
    elif method == "diff":
        return DiffSpanRepr(input_dim, use_proj=use_proj, proj_dim=proj_dim)
    elif method == "diff_sum":
        return DiffSumSpanRepr(input_dim, use_proj=use_proj, proj_dim=proj_dim)
    elif method == "endpoint":
        return EndPointRepr(input_dim, use_proj=use_proj, proj_dim=proj_dim)
    elif method == "coherent":
        return CoherentSpanRepr(input_dim, use_proj=use_proj, proj_dim=proj_dim)
    elif method == "coherent_original":
        return CoherentOrigSpanRepr(input_dim, use_proj=use_proj, proj_dim=proj_dim)
    elif method == "attn":
        return AttnSpanRepr(input_dim, use_proj=use_proj, proj_dim=proj_dim)
    elif method == "coref":
        return AttnSpanRepr(input_dim, use_proj=use_proj, proj_dim=proj_dim, use_endpoints=True)
    else:
        raise NotImplementedError

class Edge_probe_model(nn.Module):
    def __init__(self, num_of_spans, num_layers, input_span_len, embedding_dim, 
                 num_classes, pool_method='max', use_proj=True, proj_dim=256, 
                 hidden_dim=256, device='cuda', normalize_layers=False, use_cross_entropy=False):
        super(Edge_probe_model, self).__init__()
        self.device = device
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.num_of_spans = num_of_spans
        self.weighing_params = nn.Parameter(torch.ones(self.num_layers))
        self.input_dim = embedding_dim * num_of_spans
        self.use_proj = use_proj
        self.proj_dim = proj_dim
        self.normalize_layers = normalize_layers

        ## Projection
        if use_proj:
            # Apply a projection layer to output of pretrained models
            # print(embedding_dim, num_layers, proj_dim)
            self.proj1 = nn.Linear(embedding_dim, proj_dim)
            if self.num_of_spans == 2:
                self.proj2 = nn.Linear(embedding_dim, proj_dim)
            # Update the input_dim
            self.input_dim = proj_dim * num_of_spans

        ## Pooling
        self.pool_method = pool_method
        input_dim = proj_dim if use_proj else embedding_dim
        self.span1_pooling_net = get_span_module(input_dim, method=pool_method).to(device)
        if self.num_of_spans == 2:
            self.span2_pooling_net = get_span_module(input_dim, method=pool_method).to(device)

        ## Classification
        label_net_list = [
            nn.Linear(self.input_dim, hidden_dim),
            nn.Tanh(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, self.num_classes)        
        ]
        if use_cross_entropy:
            self.training_criterion = nn.CrossEntropyLoss()
        else:
            self.training_criterion = nn.BCELoss()
            label_net_list.append(nn.Sigmoid())

        self.label_net = nn.Sequential(*label_net_list)
        self.optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE, weight_decay=0)

    def forward(self, spans_torch_dict):
        span1_reprs = spans_torch_dict["span1"]
        span1_attention_mask = spans_torch_dict["span1_attention_mask"]
        if self.num_of_spans == 2:
            span2_reprs = spans_torch_dict["span2"]
            span2_attention_mask = spans_torch_dict["span2_attention_mask"]
        # print(span1_reprs.shape)
        
        ## Projection
        if self.use_proj:
            span1_reprs = self.proj1(span1_reprs)
            if self.num_of_spans == 2:
                span2_reprs = self.proj2(span2_reprs)
        
        ## Pooling
        pooled_span1 = self.span1_pooling_net(span1_reprs, span1_attention_mask)
        if self.num_of_spans == 2:
            pooled_span2 = self.span2_pooling_net(span2_reprs, span2_attention_mask)

        # print(my_dataset_handler.tokenized_dataset["train"][0])
        # print("SPAN1", span1_reprs[2, :, :, 0:5])
        # print("SPAN2", span2_reprs[2, :, :, 0:5])
        # print("MAX1", pooled_span1[2, :, 0:5])
        # print("MAX2", pooled_span2[2, :, 0:5])
        # raise "E"
        if self.normalize_layers:
            pooled_span1 = torch.nn.functional.normalize(pooled_span1, dim=-1)
            if self.num_of_spans == 2:
                pooled_span2 = torch.nn.functional.normalize(pooled_span2, dim=-1)

        if self.num_of_spans == 2:
            output = torch.cat((pooled_span1, pooled_span2), dim=-1)
        elif self.num_of_spans == 1:
            output = pooled_span1
        # print(output.shape)  # torch.Size([32, 13, 512])

        ## Mixing Weights
        wtd_encoded_repr = 0
        soft_weight = nn.functional.softmax(self.weighing_params, dim=0)
        for i in range(self.num_layers):
            # print(i, output[:, i, :].shape, torch.norm(output[:, i, :]), torch.norm(s1))
            # print(output[:, i, :][0, 0:10])
            # print(s1[0, 0:10])
            wtd_encoded_repr += soft_weight[i] * output[:, i, :]
        # wtd_encoded_repr += soft_weight[-1] * encoded_layers[:, -1, :]
        output = wtd_encoded_repr

        ## Classification
        pred_label = self.label_net(output)
        pred_label = torch.squeeze(pred_label, dim=-1)
        return pred_label

    def summary(self, do_print=True):
        summary_str = str(self)
        pytorch_total_params = sum(p.numel() for p in self.parameters())
        pytorch_total_params_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        summary_str += f"\n Total Parameters:     {pytorch_total_params}"
        summary_str += f"\n Trainable Parameters: {pytorch_total_params_trainable}"
        summary_str += f"\n Pool Method: {self.pool_method}"
        summary_str += f"\n Projection: {self.use_proj}, {self.proj_dim}"
        summary_str += f"\n normalize_layers: {self.normalize_layers}"
        if do_print:
            print(summary_str)
        return summary_str
        # print("Total Parameters:    ", pytorch_total_params)
        # print("Trainable Parameters:", pytorch_total_params_trainable)
        # print("Pool Method:", self.pool_method)
        # print("Projection:", self.use_proj, self.proj_dim)

gpu_cache = dict()
ram_cache = dict()
class Trainer(ABC):
    """ Abstract Trainer Class """
    def span_dict_to_device(self, spans_torch_dict, device="cuda"):
        new_dict = copy.deepcopy(spans_torch_dict)
        new_dict["span1"] = new_dict["span1"].to(device)
        new_dict["span1_attention_mask"] = new_dict["span1_attention_mask"].to(device)
        if self.num_of_spans == 2:
            new_dict["span2"] = new_dict["span2"].to(device)
            new_dict["span2_attention_mask"] = new_dict["span2_attention_mask"].to(device)
        return new_dict

    def prepare_batch_data(self, tokenized_dataset, start_idx, end_idx, pad=False, cache_prefix=None):
        # self.vprint("Extracting From Model")
        if cache_prefix is not None:
            cache_id = f"{cache_prefix}{start_idx}-{end_idx}"
            if cache_id in gpu_cache:
                return gpu_cache[cache_id]
            if cache_id in ram_cache:
                return self.span_dict_to_device(ram_cache[cache_id], "cuda")

        span_representations_dict = self.extract_embeddings(tokenized_dataset, start_idx, end_idx, pad=True)
        # self.vprint("To Device")
        span1_torch = torch.stack(span_representations_dict["span1"]).float().to(self.MLP_device)  # (batch_size, #layers, max_span_len, embd_dim)
        span1_attention_mask_torch = torch.stack(span_representations_dict["span1_attention_mask"])
        one_hot_labels_torch = torch.tensor(np.array(span_representations_dict["one_hot_label"]))
        if self.num_of_spans == 2:
            span2_torch = torch.stack(span_representations_dict["span2"]).float().to(self.MLP_device)
            span2_attention_mask_torch = torch.stack(span_representations_dict["span1_attention_mask"])
            spans_torch_dict = {"span1": span1_torch, 
                                "span2": span2_torch, 
                                "span1_attention_mask": span1_attention_mask_torch, 
                                "span2_attention_mask": span2_attention_mask_torch, 
                                "one_hot_labels": one_hot_labels_torch}
        elif self.num_of_spans == 1:
            spans_torch_dict = {"span1": span1_torch, 
                                "span1_attention_mask": span1_attention_mask_torch, 
                                "one_hot_labels": one_hot_labels_torch}

        if cache_prefix is not None:
            if len(gpu_cache) < GPU_CACHE_LEN:
                gpu_cache[cache_id] = spans_torch_dict
                print(cache_id, end="|")
            elif len(ram_cache) < RAM_CACHE_LEN:
                ram_cache[cache_id] = self.span_dict_to_device(spans_torch_dict, "cpu")
                print(cache_id, end=",")
        return spans_torch_dict

    def get_language_model_properties(self):
        span_representations_dict = self.extract_embeddings(self.dataset_handler.tokenized_dataset["train"], 0, 3, pad=True)
        for i in span_representations_dict["span1"]:
            print(i.shape)
        span1_torch = span_representations_dict["span1"]
        num_layers = span1_torch[0].shape[0]
        span_len = span1_torch[0].shape[1]
        embedding_dim = span1_torch[0].shape[2]
        # if self.verbose:
        #     display(pd.DataFrame(span_representations_dict))
        return num_layers, span_len, embedding_dim, len(self.dataset_handler.labels_list)

    def pad_span(self, span_repr, max_len):
        """ pad spans in embeddings to max_len 
        input:
            span_representation: df with shape (#layers, span_len, embedding_dim)
        returns:
            padded_spans: np with shape (batch_len, num_layers, max_len, embedding_dim)
            attention_mask: np with shape (max_len), values = 1: data, 0: padding
        """
        shape = span_repr.shape
        num_layers = shape[0]
        span_original_len = shape[1]
        embedding_dim = shape[2]
        # padded_span_repr = np.zeros((num_layers, max_len, embedding_dim))
        # if span_original_len > max_len:
        #     raise Exception(f"Error: {span_original_len} is more than max_span_len {max_len}\n{span_repr.shape}")

        # attention_mask = torch.tensor(np.array([1] * span_original_len + [0] * (max_len - span_original_len)), dtype=torch.int8, device=self.device)
        attention_mask = torch.ones(max_len, dtype=torch.int8, device=self.device)
        attention_mask[span_original_len:] = 0

        padded_span_repr = torch.cat((span_repr, torch.zeros((num_layers, max_len - span_original_len, embedding_dim), device=self.device)), axis=1)
        # assert attention_mask.shape == (max_len, ), f"{attention_mask}, {attention_mask.shape} != ({max_len}, )"
        # assert padded_span_repr.shape == (num_layers, max_len, embedding_dim)
        return padded_span_repr, attention_mask

    def init_span_dict(self, num_of_spans, pad):
        if num_of_spans == 2:
            span_repr = {"span1": [], "span2": [], "label": [], "one_hot_label": []}
        else:
            span_repr = {"span1": [], "label": [], "one_hot_label": []}
        
        if pad:
            span_repr["span1_attention_mask"] = []
            span_repr["span2_attention_mask"] = []
        return span_repr

    def extract_glove(self, tokenized_dataset, idx, span_start, span_end):
        text = tokenized_dataset[idx]["text"]
        embedding_dim = word_embedding.word_vectors.shape[-1]
        span_len = span_end - span_start
        hidden_states = torch.zeros(1, span_len, embedding_dim, device=self.device)  #(layers, span_len, embedding_dim)
        # print(text[0:3])
        for i in range(span_len):
            word = text[span_start + i]
            if word in word_embedding.dictionary:
                hidden_states[0, i, :] = torch.tensor(word_embedding.word_vectors[word_embedding.dictionary[word]], device=self.device)
            else:
                pass
                # print("UNKONW WORD:", word)
        return hidden_states

    def extract_elmo(self, tokenized_dataset, idx, span_start, span_end):
        text = " ".join(tokenized_dataset[idx]["text"])
        hidden_states = elmo.get_elmo_embedding(text)
        return hidden_states[:, span_start:span_end, :]


    def extract_batch(self, tokenized_dataset, idx, unique_batch_size=32):
        # if "glove" in model_checkpoint:
        #     return extract_batch_glove(tokenized_dataset, idx, unique_batch_size)
        # if "elmo" in model_checkpoint:
        #     return extract_batch_elmo(tokenized_dataset, idx, unique_batch_size)

        # print(idx)
        self.vprint("e1")
        dataset_len = len(tokenized_dataset)
        unique_texts_in_batch = []
        i = idx
        while len(unique_texts_in_batch) < unique_batch_size and i < dataset_len:
            # print(i)
            text = tokenized_dataset[i]["text"]
            if not text in unique_texts_in_batch:
                unique_texts_in_batch.append(text)
            i += 1
        tokenizer.padding_side = 'right'  # Important: lef will change the span indices
        tokenized_batch = tokenizer(unique_texts_in_batch, padding=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            if SEQ2SEQ_MODEL:
                outputs = self.language_model(input_ids=tokenized_batch.input_ids, decoder_input_ids=tokenized_batch.input_ids, output_hidden_states=True)
            else:
                outputs = self.language_model(**tokenized_batch)
        # torch.cuda.synchronize()
        # current_hidden_states = np.asarray([val.detach().cpu().numpy() for val in outputs.hidden_states])
        if SEQ2SEQ_MODEL:
            encoder_hidden_states = torch.stack([val.detach() for val in outputs.encoder_hidden_states])
            decoder_hidden_states = torch.stack([val.detach() for val in outputs.decoder_hidden_states])
            current_hidden_states = torch.cat((encoder_hidden_states, decoder_hidden_states), dim=0)  # concat from layers
        else:
            current_hidden_states = torch.stack([val.detach() for val in outputs.hidden_states])
        # self.vprint(current_hidden_states.shape)  # (13, 16, 34, 768)
        
        extracted_batch_embeddings = {}
        for i, unique_text in enumerate(unique_texts_in_batch):
            hashable_input = repr(unique_text)
            if not hasattr(self, 'up_to_layer') or self.up_to_layer == -1:
                extracted_batch_embeddings[hashable_input] = current_hidden_states[:, i, :, :]
            else:
                extracted_batch_embeddings[hashable_input] = current_hidden_states[:self.up_to_layer+1, i, :, :]
        self.vprint("e2")
        return extracted_batch_embeddings
    
    def pad_sequence(list_of_torch, pad_len, pad_value=0):
        shape = list_of_torch[0].shape
        num_layers = shape[0]
        span_original_len = shape[1]
        embedding_dim = shape[2]
        output = torch.zeros()

    def extract_embeddings(self, tokenized_dataset, start_idx, end_idx, pad=True):
        """ Extract raw embeddings for [start_idx, end_idx) of tokenized_dataset from language_model 
            
        Returns:
            extract_embeddings: DataFrame with cols (span1, span2?, label) and span shape is (range_len, (#layers, span_len, embedding_dim))
        """
        num_of_spans = self.dataset_handler.dataset_info.num_of_spans
        
        if num_of_spans == 2:
            max_span_len_in_batch = max(max(tokenized_dataset[start_idx:end_idx]["span1_len"]), max(tokenized_dataset[start_idx:end_idx]["span2_len"]))
        elif num_of_spans == 1:
            max_span_len_in_batch = max(tokenized_dataset[start_idx:end_idx]["span1_len"])
        # print("max_span_len_in_batch", max_span_len_in_batch)
        

        span_repr = self.init_span_dict(num_of_spans, pad)
        self.vprint("f1")
        for i in range(start_idx, end_idx):
            row = tokenized_dataset[i]
            if "glove" in model_checkpoint:
                span1_hidden_states = self.extract_glove(tokenized_dataset, i, row["span1"][0], row["span1"][1])
            elif "elmo" in model_checkpoint:
                span1_hidden_states = self.extract_elmo(tokenized_dataset, i, row["span1"][0], row["span1"][1])
                self.vprint("f2")
            else:
                hashable_input = repr(tokenized_dataset[i]["text"])
                
                if hashable_input in self.cached_embeddings:
                    self.current_hidden_states = self.cached_embeddings[hashable_input]
                else:
                    if hashable_input not in self.extracted_batch_embeddings:
                        self.extracted_batch_embeddings = self.extract_batch(tokenized_dataset, i)
                        # if len(self.cached_embeddings) < CACHE_LEN:
                        #     for key, value in self.extracted_batch_embeddings.items():
                        #         self.cached_embeddings[key] = value
                        #     print(f"Cached {len(self.cached_embeddings)}")
                    self.current_hidden_states = self.extracted_batch_embeddings[hashable_input]
                
                # if hashable_input not in self.extracted_batch_embeddings:
                #     self.extracted_batch_embeddings = self.extract_batch(tokenized_dataset, i)    
                # self.current_hidden_states = self.extracted_batch_embeddings[hashable_input]

                span1_hidden_states = self.current_hidden_states[:, row["span1"][0]:row["span1"][1], :]  # (#layer, span_len, embd_dim)
            
            if pad:
                s1, a1 = self.pad_span(span1_hidden_states, max_span_len_in_batch)
                span_repr["span1"].append(s1)
                span_repr["span1_attention_mask"].append(a1)
            else:
                span_repr["span1"].append(span1_hidden_states)

            if num_of_spans == 2:
                if "glove" in model_checkpoint:
                    span2_hidden_states = self.extract_glove(tokenized_dataset, i, row["span2"][0], row["span2"][1])
                elif "elmo" in model_checkpoint:
                    span2_hidden_states = self.extract_elmo(tokenized_dataset, i, row["span2"][0], row["span2"][1])
                    self.vprint("f3")
                else:
                    span2_hidden_states = self.current_hidden_states[:, row["span2"][0]:row["span2"][1], :]
                if pad:
                    s2, a2 = self.pad_span(span2_hidden_states, max_span_len_in_batch)
                    span_repr["span2"].append(s2)
                    span_repr["span2_attention_mask"].append(a2)
                else:
                    span_repr["span2"].append(span2_hidden_states)
            span_repr["one_hot_label"].append(row["one_hot_label"])
            span_repr["label"].append(row["label"])
        self.vprint("f4")
        return span_repr

    def save_history(self, history_dict, mdl=False):
        if mdl == True:
            prefix = "mdl/mdl_results_"
            history_dict = {"mdl_history": history_dict}
        else:
            prefix = "edge_probing_results/"
        file_name = prefix + model_checkpoint + "_" + self.dataset_handler.dataset_info.dataset_name + "_" + str(SEED)
        history_dict["Model"] = model_checkpoint,
        history_dict["Batch Size"] = BATCH_SIZE,
        history_dict["Learning Rate"] = LEARNING_RATE,
        history_dict["seed"] = SEED
        if hasattr(self, 'edge_probe_model'):
            history_dict["probe_summary"] = self.edge_probe_model.summary(do_print=False)
        elif hasattr(self, 'edge_probe_models'):
            history_dict["probe_summary"] = self.edge_probe_models[0].summary(do_print=False)
        else:
            print("No Probe Found to Summarize!")
        history_dict["dataset_name"] = self.dataset_handler.dataset_info.dataset_name
        history_dict["dataset_statistics"] = str(self.dataset_handler.dataset)

        from pathlib import Path
        Path(file_name).mkdir(parents=True, exist_ok=True)
        with open(f"{file_name}.json", "w") as json_file:
            json.dump(history_dict, json_file, indent=4)
        # with open(f"{file_name}.pkl", "wb") as pkl_file:
        #     pickle.dump(history_dict, pkl_file)

"""# Edge Probe Trainer"""

class Edge_probe_trainer(Trainer):
    # Public:
    def __init__(self, language_model, dataset_handler: Dataset_handler, 
                 verbose=True, device='cuda', edge_probe_model_checkpoint=None,
                 pool_method="attn", start_eval = False, 
                 history_checkpoint=None, up_to_layer=-1, normalize_layers=False):
        self.dataset_handler = dataset_handler
        self.num_of_spans = self.dataset_handler.dataset_info.num_of_spans
        self.up_to_layer = up_to_layer
        self.language_model = language_model
        self.language_model.config.output_hidden_states = True
        self.device = device
        self.verbose = verbose
        self.start_eval = start_eval
        def vprint(text):
            if verbose:
                print(datetime.datetime.now().time(), text)
        self.vprint = vprint

        self.current_hidden_states = None
        self.last_input_ids = None
        self.extracted_batch_embeddings = {}

        self.cached_embeddings = {}

        self.vprint("Moving to device")
        for param in self.language_model.parameters():
            param.requires_grad = False
        self.language_model.eval()
        self.language_model.to(self.device)
        num_layers, input_span_len, embedding_dim, num_classes = self.get_language_model_properties()
        self.MLP_device = self.device
        if edge_probe_model_checkpoint == None:
            print("Creating New EPM")
            self.edge_probe_model = Edge_probe_model(
                num_of_spans = self.num_of_spans,
                num_layers = num_layers,
                input_span_len = input_span_len,
                embedding_dim = embedding_dim, 
                num_classes = num_classes,
                device = self.MLP_device,
                pool_method = pool_method,
                normalize_layers = normalize_layers
            )
        else:
            print("Starting From a Pretrained EPM")
            self.edge_probe_model = edge_probe_model_checkpoint
        

        if history_checkpoint is None:
            self.history = {"loss": {"train": [], "dev": [], "test": []}, 
                            "metrics": 
                            {
                                "micro_f1": {"dev": [], "test": []},
                                "macro_f1": {"dev": [], "test": []},
                                "accuracy": {"dev": [], "test": []},
                                "report": {"dev": [], "test": []}
                            },

                            "layers_weights": [],
                            }
            print("Creating New History")
        else:
            print("Using History Checkpoint")
            self.history = history_checkpoint
    
    def train(self, batch_size, epochs=3):
        tokenized_dataset = self.dataset_handler.tokenized_dataset["train"]
        tokenized_dataset_dev = self.dataset_handler.tokenized_dataset["dev"]
        tokenized_dataset_test = self.dataset_handler.tokenized_dataset["test"]

        # self.edge_probe_model.to(self.device)
        self.edge_probe_model.to(self.MLP_device)
        # self.vprint("Counting dataset rows")
        dataset_len = len(tokenized_dataset)
        dev_dataset_len = len(tokenized_dataset_dev)
        test_dataset_len = len(tokenized_dataset_test)
        print(f"Train on {dataset_len} samples, validate on {dev_dataset_len} samples, test on {test_dataset_len} samples")
        # dataset_len = 60
        if self.start_eval:
            self.update_history(epoch = 0)

        step_counter = 0
        for epoch in range(epochs):
            running_loss = 0.0
            steps = 0
            print("----------------\n")
            self.edge_probe_model.train()
            for i in tqdm(range(0, dataset_len, batch_size), desc=f"[Epoch {epoch + 1}/{epochs}]"):
                
                # step_counter += 1
                # if i > 0 and step_counter > 0 and step_counter % 1000 == 0:
                #     self.update_history(epoch + 1, train_loss = running_loss / steps)
                #     self.draw_weights(epoch)
                
                self.vprint("Start")
                step = batch_size
                if i + batch_size > dataset_len:
                    step = dataset_len - i
                # print(f"WWW[{i}, {i+step})")
                
                self.vprint("Extracting")
                # self.vprint("prepare")
                spans_torch_dict = self.prepare_batch_data(tokenized_dataset, i, i + step, pad=True, cache_prefix="t")
                labels = spans_torch_dict["one_hot_labels"]
                # zero the parameter gradients
                self.edge_probe_model.optimizer.zero_grad()
    
                # forward + backward + optimize
                self.vprint("Forward MLP")
                # self.vprint("epm")
                outputs = self.edge_probe_model(spans_torch_dict)
                self.vprint("Loss")
                loss = self.edge_probe_model.training_criterion(outputs.to(self.device), labels.float().to(self.device))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.edge_probe_model.parameters(), 5.0)
                self.edge_probe_model.optimizer.step()
    
                running_loss += loss.item()
                steps += 1
                self.vprint("Done")
                
                
                if i/batch_size % 50 == 0:
                    self.update_history(epoch + 1, train_loss = running_loss / steps)
                    # self.draw_weights(epoch)
                
                # print(f"loss: {running_loss / steps}")

            self.update_history(epoch + 1, train_loss = running_loss / steps)
            # self.draw_weights(epoch)
            self.save_history(self.history)

    def calc_loss(self, tokenized_dataset, batch_size=16, print_metrics=False, just_micro=False, desc=""):
        self.edge_probe_model.eval()
        with torch.no_grad():
            running_loss = 0
            dataset_len = len(tokenized_dataset["text"])
            steps = 0
            preds = None
            for i in tqdm(range(0, dataset_len, batch_size), desc=desc):
                # if int(i / batch_size) % 100 == 0:
                #     print("memory:", psutil.virtual_memory().percent, gc.collect(), psutil.virtual_memory().percent)
                step = batch_size
                if i + batch_size > dataset_len:
                    step = dataset_len - i
                
                if desc == "Test Loss":
                    spans_torch_dict = self.prepare_batch_data(tokenized_dataset, i, i + step, pad=True, cache_prefix="test")
                else:
                    spans_torch_dict = self.prepare_batch_data(tokenized_dataset, i, i + step, pad=True)
                labels = spans_torch_dict["one_hot_labels"]
                # forward
                outputs = self.edge_probe_model(spans_torch_dict)
                
                preds = outputs if i == 0 else torch.cat((preds, outputs), 0)
                loss = self.edge_probe_model.training_criterion(outputs.to(self.device), labels.float().to(self.device))
                running_loss += loss.item()
                steps += 1

        # print(preds[0:4])
        preds = preds.cpu().argmax(-1)
        y_true = np.array(tokenized_dataset["one_hot_label"]).argmax(-1)
        # print(preds[0:4])
        # print(y_true[0:4])
        labels_list = self.dataset_handler.labels_list
        if self.dataset_handler.dataset_info.dataset_name == "semeval":
            other_idx = labels_list.index("Other")
            semeval_labels = [i for i in range(len(labels_list))]
            semeval_labels.remove(other_idx)
            print(f"Other -> {other_idx} / {semeval_labels}")
            micro_f1 = f1_score(y_true, preds, average='micro', labels=semeval_labels)
            macro_f1 = f1_score(y_true, preds, average='macro', labels=semeval_labels)
        else:
            micro_f1 = f1_score(y_true, preds, average='micro')
            macro_f1 = f1_score(y_true, preds, average='macro')
        
        accuracy = sklearn.metrics.accuracy_score(y_true, preds)
        
        
        report = classification_report(y_true, preds, target_names=labels_list, labels=range(len(labels_list)))
        if print_metrics:
            if not just_micro and desc == "Test Loss":
                print(report)
            print("MICRO F1:", micro_f1)
        return running_loss / steps, micro_f1, macro_f1, accuracy, report

    # Private:
    def update_history(self, epoch, train_loss = None):
        if train_loss is None:
            train_loss, train_f1 = self.calc_loss(self.dataset_handler.tokenized_dataset["train"], print_metrics=True, desc="Train Loss")
        dev_loss, dev_f1, dev_macro_f1, dev_accuracy, dev_report = self.calc_loss(self.dataset_handler.tokenized_dataset["dev"], print_metrics=True, desc="Dev Loss")
        test_loss, test_f1, test_macro_f1, test_accuracy, test_report = self.calc_loss(self.dataset_handler.tokenized_dataset["test"], print_metrics=True, desc="Test Loss")
        self.history["loss"]["train"].append(train_loss)
        self.history["loss"]["dev"].append(dev_loss)
        self.history["loss"]["test"].append(test_loss)

        self.history["metrics"]["micro_f1"]["dev"].append(dev_f1)
        #self.history["metrics"]["macro_f1"]["dev"].append(dev_macro_f1)
        self.history["metrics"]["accuracy"]["dev"].append(dev_accuracy)
        # self.history["metrics"]["report"]["dev"].append(dev_report)

        self.history["metrics"]["micro_f1"]["test"].append(test_f1)
        #self.history["metrics"]["macro_f1"]["test"].append(test_macro_f1)
        self.history["metrics"]["accuracy"]["test"].append(test_accuracy)
        # self.history["metrics"]["report"]["test"].append(test_report)

        self.history["layers_weights"].append(torch.nn.functional.softmax(self.edge_probe_model.weighing_params).tolist())
        print('[%d] loss: %.4f, val_loss: %.4f, test_loss: %.4f' % (epoch, self.history["loss"]["train"][-1], self.history["loss"]["dev"][-1], self.history["loss"]["test"][-1]))

    def draw_weights(self, epoch=0):
        if(epoch % 1 == 0):
            # w = self.edge_probe_model.weighing_params.tolist()
            w = torch.nn.functional.softmax(self.edge_probe_model.weighing_params).cpu().detach().numpy()
            print(w)
            # print(self.history)
            plt.bar(np.arange(len(w), dtype=int), w)
            plt.ylabel('Weight')
            plt.xlabel('Layer');
            plt.show()

            wsoft = nn.functional.softmax(self.edge_probe_model.weighing_params)
            print("CG", sum(idx*val for idx, val in enumerate(wsoft)))

            print("Loss History")
            loss_history = self.history["loss"]
            x = range(len(loss_history["train"]))
            plt.plot(x, loss_history["train"])
            plt.plot(x, loss_history["dev"])
            plt.plot(x, loss_history["test"])
            plt.legend(['Train', 'Dev', 'Test'], loc='lower left')
            plt.show()

            print("Micro f1 History")
            f1_history = self.history["metrics"]["micro_f1"]
            x = range(len(f1_history["dev"]))
            plt.plot(x, f1_history["dev"])
            plt.plot(x, f1_history["test"])
            plt.legend(['Dev', 'Test'], loc='upper left')
            plt.show()

    def analyze_attention(self, dataset_part="train", row_idx=0, new_text = None):
        if new_text is not None:
            new_dataset_info = Dataset_info("manual", num_of_spans=1, manual_text=new_text)
            new_dataset_handler = Dataset_handler(new_dataset_info);
            dataset_part = "test"
            row_idx=0
            tokenized_dataset = new_dataset_handler.tokenized_dataset[dataset_part]
        else:
            tokenized_dataset = self.dataset_handler.tokenized_dataset[dataset_part]
        
        self.edge_probe_model.eval()
        with torch.no_grad():
            spans_torch_dict = self.prepare_batch_data(tokenized_dataset, row_idx, row_idx + 1, pad=True)
            labels = spans_torch_dict["one_hot_labels"]
            # forward
            outputs = self.edge_probe_model(spans_torch_dict)
        preds = outputs
        loss = self.edge_probe_model.training_criterion(outputs.to(self.device), labels.float().to(self.device))
        running_loss = loss.item()
        print(preds[0:9])
        preds = preds.cpu().argmax(-1)
        y_true = np.array(tokenized_dataset["one_hot_label"]).argmax(-1)
        # print(preds[0:9])
        # print(y_true[row_idx])
        if "glove" in model_checkpoint:
            text = tokenized_dataset[row_idx]["text"]
        else:
            start = tokenized_dataset[row_idx]["span1"][0]
            end = tokenized_dataset[row_idx]["span1"][1]
            text = tokenizer.convert_ids_to_tokens(tokenized_dataset[row_idx]["input_ids"][start:end])
        return text, self.edge_probe_model.span1_pooling_net.last_attention_wts, preds, y_true[row_idx]

my_edge_probe_trainer = None
edge_probe_model_checkpoint = None
history = None
gc.collect()
torch.cuda.empty_cache()

try:
    edge_probe_model_checkpoint = my_edge_probe_trainer.edge_probe_model
except:
    edge_probe_model_checkpoint = None
my_edge_probe_trainer = Edge_probe_trainer(model,
                                           my_dataset_handler, 
                                           device=DEVICE,
                                           pool_method=POOL_METHOD,
                                           edge_probe_model_checkpoint=edge_probe_model_checkpoint,
                                           history_checkpoint=history,
                                           up_to_layer = -1,
                                           normalize_layers=True,
                                           verbose=False)

print("Model:", model_checkpoint)
print("Dataset:", my_dataset_info.dataset_name)
print(f"Batch Size: {BATCH_SIZE}")
a = my_edge_probe_trainer.edge_probe_model.summary(do_print=True)

my_edge_probe_trainer.train(batch_size = BATCH_SIZE, epochs=30)
