
# Metaphors in Pre-Trained Language Models: Probing and Generalization Across Datasets and Languages

_Accepted as a conference paper for ACL 2022_

[Arxiv](https://arxiv.org/abs/2203.14139)

> **Abstract**: Human languages are full of metaphorical expressions. Metaphors help people understand the world by connecting new concepts and domains to more familiar ones. Large pre-trained language models (PLMs) are therefore assumed to encode metaphorical knowledge useful for NLP systems. In this paper, we investigate this hypothesis for PLMs, by probing metaphoricity information in their encodings, and by measuring the cross-lingual and cross-dataset generalization of this information. We present studies in multiple metaphor detection datasets and in four languages (i.e., English, Spanish, Russian, and Farsi). Our extensive experiments suggest that contextual representations in PLMs do encode metaphorical knowledge, and mostly in their middle layers. The knowledge is transferable between languages and datasets, especially when the annotation is consistent across training and testing sets. Our findings give helpful insights for both cognitive and NLP scientists.


## Requirements

### Software
Please find required libraries in the `requirements.txt` file. You can install them by using the package manager [pip](https://pip.pypa.io/en/stable/).
```bash
pip install -r requirements.txt
```

## Run Edge Probing
You can run the edge probing by running the following command:
```
python3 EDGE_CODE_PATH MODEL_NAME TASK SEED
```
Example:
```
python3 source_code/scripts/edge_probing.py bert-base-uncased trofi 0
```
MODEL_NAME:
```
bert-base-uncased
roberta-base
google/electra-base-discriminator
xlm-roberta-base
bert-base-uncased-random-weights
xlm-roberta-base-random-weights
```
TASK_NAME:
```
lcc
trofi
vua_verb
vua_pos
lcc_fa
lcc_es
lcc_ru
lcc_trofi
trofi_lcc
lcc-vua_pos
vua_pos-lcc
lcc-vua_verb
vua_verb-lcc
trofi-vua_pos
vua_pos-trofi
trofi-vua_verb
vua_verb-trofi
vua_pos-vua_verb
vua_verb-vua_pos
lcc-lcc
trofi-trofi
vua_pos-vua_pos
vua_verb-vua_verb
```

## Run MDL Probing
You can run the MDL probing by running the following command (same as the edge probing):
```
python3 MDL_CODE_PATH MODEL_NAME TASK SEED
```
Example:
```
python3 source_code/scripts/mdl_probing.py bert-base-uncased trofi 0
```