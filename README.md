
# Metaphors in Pre-Trained Language Models: <br> Probing and Generalization Across Datasets and Languages

_Accepted as a conference paper at ACL 2022_

[📝 Arxiv](https://arxiv.org/abs/2203.14139)
[🎥 Video](https://www.youtube.com/watch?v=UKWFZSiP7OY)
[🖼️ Poster](https://mohsenfayyaz.github.io/files/publications/2022_metaphors_in_plms/metaphors_poster_36x48.pdf)


> **Abstract**: Human languages are full of metaphorical expressions. Metaphors help people understand the world by connecting new concepts and domains to more familiar ones. Large pre-trained language models (PLMs) are therefore assumed to encode metaphorical knowledge useful for NLP systems. In this paper, we investigate this hypothesis for PLMs, by probing metaphoricity information in their encodings, and by measuring the cross-lingual and cross-dataset generalization of this information. We present studies in multiple metaphor detection datasets and in four languages (i.e., English, Spanish, Russian, and Farsi). Our extensive experiments suggest that contextual representations in PLMs do encode metaphorical knowledge, and mostly in their middle layers. The knowledge is transferable between languages and datasets, especially when the annotation is consistent across training and testing sets. Our findings give helpful insights for both cognitive and NLP scientists.



## Running Probings
An online colab notebook is available at [Metaphor_Demo.ipynb](Metaphor_Demo.ipynb)

You can run the probings by running the following command:
```
python3 {EDGE_CODE_PATH/MDL_CODE_PATH} {MODEL_NAME} {TASK_NAME} {SEED}
```
Example:
```
python3 source_code/scripts/edge_probing.py bert-base-uncased trofi 0
python3 source_code/scripts/mdl_probing.py bert-base-uncased trofi 0
```
MODEL_NAME:
```
bert-base-uncased
roberta-base
google/electra-base-discriminator
xlm-roberta-base
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
```
