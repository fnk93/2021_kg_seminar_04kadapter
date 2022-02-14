# KIT Seminar Representation Learning on Knowledge Graphs: K-Adapter

In this repository, I tried to make the results of the paper reproducible.
Some datasets are already included in this repository, where others are too big to upload to github.
I created two docker containers ([without Cuda](https://hub.docker.com/repository/docker/fnk93/2021kgseminar04kadapter), [with Cuda](https://hub.docker.com/repository/docker/fnk93/2021kgseminar04kadaptercuda)) which are prepared with all needed datasets as well as the pre-trained adapters of the original authors.

When using the docker container including Cuda, make sure to first install [NVIDIA CUDA v11.3.1 or later](https://developer.nvidia.com/cuda-toolkit-archive) as well as the [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker)

and start the container with:

    docker run --rm -it --init --gpus=all fnk93/2021kgseminar04kadaptercuda:latest

If you want to clone the repository and install dependencies (using Python 3.8 or newer), you can do so by:

    git clone https://github.com/fnk93/2021_kg_seminar_04kadapter.git
    cd 2021_kg_seminar_04_kadapter
    python -m pip install --upgrade -r requirements.txt

Hyperparameters in the run scripts are set to work on a single GPU instance.
Best hyperparameters according to the paper are listed in the run scripts as well. Mostly only the batch size had to be reduced.

If you want to use the run scripts, either remove the `--save_to_s3` and `--read_from_s3` arguments, or make sure to set `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` and `AWS_DEFAULT_REGION` according to your aws account and create a bucket in S3 called _kadapter_.
These flags allow you to read the datasets from _s3://kadapter/data_ and save the results to _s3://kadapter/results/_ where the results will be put into different folders in regards to the adapters used (_F_, _L_ or _F+L_) as well as the dataset used.

If you want to be able to resume fine-tuning after shutting down, make sure to add the `--restore` argument to your run script, this will look for saved checkpoints in the respective folder under _/proc_data_.

## Pre-trained Adapters

[Google Cloud](https://drive.google.com/drive/folders/12mfLpYq4BTwdbLZnQsdwDJKubM9aOr9f)

Pre-trained adapters have been obtained from the google drive link of the original authors.

## Datasets

[Open Entity](https://www.cs.utexas.edu/~eunsol/html_pages/open_entity.html)

[FIGER](https://drive.google.com/open?id=0B52yRXcdpG6MMnRNV3dTdGdYQ2M)

[TACRED](https://drive.google.com/open?id=1HlWw7Q6-dFSm9jNSCh4VaBf1PlGqt9im)

For _TACRED_ the data of the url provided had to be transformed, see [_/scripts/convert_tacred.py_](https://github.com/fnk93/2021_kg_seminar_04kadapter/blob/main/scripts/convert_tacred.py)

### SearchQA, CosmosQA, Quasar-T

[CosmosQA](https://storage.googleapis.com/ai2-mosaic/public/cosmosqa/cosmosqa-data.zip)

[SearchQA + Quasar-T](https://thunlp.oss-cn-qingdao.aliyuncs.com/OpenQA_data.tar.gz)

[SearchQA](https://drive.google.com/drive/u/0/folders/1kBkQGooNyG0h8waaOJpgdGtOnlb1S649)

[Quasar-T](http://curtis.ml.cmu.edu/datasets/quasar/)

[Quasar-T Background Corpus](https://lemurproject.org/clueweb09/)

The datasets _SearchQA_ and _Quasar-T_ are included in both docker containers, but since the authors didn't include their code, no own fine-tuning tests have been done.

## New datasets

Additionally, entity typing and relation classification tasks have been done on _LiterallyWikidata_. As well as relation classification tasks on _WN18RR_ and _FB15K-237_.

[LitWD1K, LitWD19K, LitWD48K](https://zenodo.org/record/4701190#.YW3sbHUza7I)

For entity typing, I pulled textual representation of entity names and entity types from WikiData and converted the dataset into the format of _FIGER_. See [_/scripts/convert_litwd.py_](https://github.com/fnk93/2021_kg_seminar_04kadapter/blob/main/scripts/convert_litwd.py) for the conversion script and [_/data/LitWD1K_](https://github.com/fnk93/2021_kg_seminar_04kadapter/tree/main/data/LitWD1K), [_/data/LitWD19K_](https://github.com/fnk93/2021_kg_seminar_04kadapter/tree/main/data/LitWD19K) and [_/data/LitWD48K_](https://github.com/fnk93/2021_kg_seminar_04kadapter/tree/main/data/LitWD48K) for the resulting data structure.

For relation classification, I pulled textual representation of entity names and relations from WikiData and converted the dataset into the format of _TACRED_. See [_/scripts/convert_litwd_rel_class.py_](https://github.com/fnk93/2021_kg_seminar_04kadapter/blob/main/scripts/convert_litwd_rel_class.py) for the conversion script and [_/data/LitWD1Krel_](https://github.com/fnk93/2021_kg_seminar_04kadapter/tree/main/data/LitWD1Krel), [_/data/LitWD19Krel_](https://github.com/fnk93/2021_kg_seminar_04kadapter/tree/main/data/LitWD19Krel) and [_/data/LitWD48Krel_](https://github.com/fnk93/2021_kg_seminar_04kadapter/tree/main/data/LitWD48Krel) for the resulting data structure.

[WN18RR](https://github.com/villmow/datasets_knowledge_embedding/tree/master/WN18RR)

For relation classification, I converted the dataset into the format of _TACRED_. See [_/scripts/convert_wn18rr_rel_class.py_](https://github.com/fnk93/2021_kg_seminar_04kadapter/blob/main/scripts/convert_wn18rr_rel_class.py) for the conversion script and [_/data/WN18RRrel_](https://github.com/fnk93/2021_kg_seminar_04kadapter/tree/main/data/WN18RRrel) for the resulting data structure. The entities and relations have not been transformed into text in this case.

[FB15K-237](https://www.microsoft.com/en-us/download/details.aspx?id=52312)

For relation classification, I converted the dataset into the format of _TACRED_. See [_/scripts/convert_fb15k_rel_class.py_](https://github.com/fnk93/2021_kg_seminar_04kadapter/blob/main/scripts/convert_fb15k_rel_class.py) for the conversion script and [_/data/FB15K-237rel_](https://github.com/fnk93/2021_kg_seminar_04kadapter/tree/main/data/FB15K-237rel) for the resulting data structure. The entities and relations have not been transformed into text in this case.

## K-Adapter: Infusing Knowledge into Pre-Trained Models with Adapters

This repository is the official implementation of the paper "K-Adapter: Infusing Knowledge into Pre-Trained Models with Adapters", ACL-IJCNLP 2021 Findings.

In the K-adapter paper, we present a flexible approach that supports continual knowledge infusion into large pre-trained models (e.g. RoBERTa in this work).
We infuse factual knowledge and linguistic knowledge, and show that adapters for both kinds of knowledge work well on downstream tasks.

For more details, please check the latest version of the paper: [https://arxiv.org/abs/2002.01808](https://arxiv.org/abs/2002.01808)

### Prerequisites 
- Python 3.6
- PyTorch 1.3.1
- tensorboardX
- transformers

We use huggingface/transformers framework, the environment can be installed with:
```bash
conda create -n kadapter python=3.6
```
```bash
pip install -r requirements.txt
```

### Pre-training Adapters
In the pre-training procedure, we train each knowledge-specific adapter on different pre-training tasks individually. 
#### 1. Process Dataset
- `./scripts/clean_T_REx.py`: clean [raw T-Rex dataset](https://hadyelsahar.github.io/t-rex/downloads/) (32G), and save the cleaned T-Rex to JSON format
- `./scripts/create_subdataset-relation-classification.ipynb`: create the dataset from T-REx for pre-training factual adapter on relation classification task. This sub-dataset can be found [here](https://drive.google.com/drive/folders/1xRGmIUXwPrtnsksQ1GY8YAE87gf7Ct6E?usp=sharing).
- `refer to this` [code](https://github.com/windweller/DisExtract/tree/master/preprocessing) to get the dependency parsing dataset : create the dataset from Book Corpus for pre-training the linguistic adapter on dependency parsing task.

#### 2. Factual Adapter
To pre-train fac-adapter, run
```bash
bash run_pretrain_fac-adapter.sh
```
#### 3. Linguistic Adapter
To pre-train lin-adapter, run
```bash
bash run_pretrain_lin-adapter.sh
```
The pre-trained fac-adapter and lin-adapter models can be found [here](https://drive.google.com/drive/folders/12mfLpYq4BTwdbLZnQsdwDJKubM9aOr9f?usp=sharing).

### Fine-tuning on Downstream Tasks
Adapter Structure
- The fac-adapter (lin-adapter) consists of two transformer layers (L=2, H=768, A = 12)
- The RoBERTa layers where adapters plug in: 0,11,23 or 0,11,22
- For using only single adapter
    - Use the concatenation of the last hidden feature of RoBERTa and the last hidden feature of the adapter as the input representation for the task-specific layer.
- For using combine adapter
    - For each adapter, first concat the last hidden feature of RoBERTa and the last hidden feature of every adapter and feed into a linear layer separately, then concat the representations as input for task-specific layer.

About how to load pretrained RoBERTa and pretrained adapter
- The pre-trained adapters are in `./pretrained_models/fac-adapter/pytorch_model.bin` and `./pretrained_models/lin-adapter/pytorch_model.bin`.
    For using only single adapter, for example, fac-adapter, then you can set the argument `meta_fac_adaptermodel=<the path of factual adapter model>` and set `meta_lin_adaptermodel=””`.
    For using both adapters, just set the arguments `meta_fac_adaptermodel` and `meta_lin_adaptermodel` as the path of adapters.
- The pretrained RoBERTa will be downloaded automaticly when you run the pipeline.

#### 1. Entity Typing
##### 1.1 OpenEntity
One single 16G P100

**(1) run the pipeline**
```bash
bash run_finetune_openentity_adapter.sh
```
**(2) result**
- with fac-adapter
    dev: (0.7967123287671233, 0.7580813347236705, 0.7769169115682607)
    test: (0.7929708951125755, 0.7584033613445378, 0.7753020134228187)
- with lin-adapter
    dev: (0.8071672354948806, 0.7398331595411888, 0.7720348204570185)
    test:(0.8001135718341851, 0.7400210084033614, 0.7688949522510232)
- with fac-adapter + lin-adapter
    dev: (0.8001101321585903, 0.7575599582898853, 0.7782538832351366)
    test: (0.7899568034557235, 0.7627737226277372, 0.7761273209549072)

the results may vary when running on different machines, but should not differ too much.
I just search results from per_gpu_train_batch_sizeh: [4, 8] lr: [1e-5, 5e-6], warmup[0,200,500,1000,1200], maybe you can change other parameters and see the results.
For w/fac-adapter, the best performance is achieved at gpu_num=1, per_gpu_train_batch_size=4, lr=5e-6, warmup=500(it takes about 2 hours to get the best result running on singe 16G P100)
For w/lin-adapter, the best performance is achieved at gpu_num=1, per_gpu_train_batch_size=4, lr=5e-6, warmup=1000(it takes about 2 hours to get the best result running on singe 16G P100)

**(3) Data format**

Add special token "@" before and after a certain entity, then the first @ is adopted to perform classification.
9 entity categories: ['entity', 'location', 'time', 'organization', 'object', 'event', 'place', 'person', 'group'], each entity can be classified to several of them or none of them. The output is represented as [0,1,1,0,1,0,0,0,0], 0 represents the entity does not belong to the type, while 1 belongs to.

##### 1.2 FIGER
**(1) run the pipeline**
```bash
bash run_finetune_figer_adapter.sh
```
The detailed hyperparamerters are listed in the running script.

#### 2. Relation Classification
4*16G P100

**(1) run the pipeline**
```bash
bash run_finetune_tacred_adapter.sh
```
**(2) result**
- with fac-adapter
    - 'dev': (0.6686945083853996, 0.7481604120676968, 0.7061989928807085)
    - 'test': (0.693900391717963, 0.7458646616541353, 0.7189447746050153)
- with lin-adapter
    - 'dev': (0.6679165308118683, 0.7536791758646063, 0.7082108902333621),
    - 'test': (0.6884615384615385, 0.7536842105263157, 0.7195979899497488)
- with fac-adapter + lin-adapter
    - 'dev': (0.6793893129770993, 0.7367549668874173, 0.7069102462271645)
    - 'test': (0.7014245014245014, 0.7404511278195489, 0.7204096561814192)

- the results may vary when running on different machines, but should not differ too much.
- I just search results from per_gpu_train_batch_sizeh: [4, 8] lr: [1e-5, 5e-6], warmup[0,200,1000,1200], maybe you can change other parameters and see the results.
- The best performance is achieved at gpu_num=4, per_gpu_train_batch_size=8, lr=1e-5, warmup=200 (it takes about 7 hours to get the best result running on 4 16G P100)
- The detailed hyperparamerters are listed in the running script.

**(3) Data format**

Add special token "@" before and after the first entity, add '#' before and after the second entity. Then the representations of  @ and # are concatenated to perform relation classification.


#### 3. Question Answering
##### 3.1 CosmosQA
One single 16G P100

**(1) run the pipeline**
```bash
bash run_finetune_cosmosqa_adapter.sh
```

**(2) result**

CosmosQA dev accuracy: 80.9
CosmosQA test accuracy: 81.8

The best performance is achieved at gpu_num=1, per_gpu_train_batch_size=64, GRADIENT_ACC=32, lr=1e-5, warmup=0 (it takes about 8 hours to get the best result running on singe 16G P100)
The detailed hyperparamerters are listed in the running script.

**(3) Data format**

For each answer, the input is `<s>context</s></s>question</s></s>answer</s>`, and will get a score for this answers. After getting four scores, we will select the answer with the highest score.

##### 3.2 SearchQA and Quasar-T 
The source codes for fine-tuning on SearchQA and Quasar-T dataset are modified based on the [code](https://github.com/thunlp/OpenQA) of paper "Denoising Distantly Supervised Open-Domain Question Answering".

### Use K-Adapter just like RoBERTa 
- You can use K-Adapter (RoBERTa with adapters) just like RoBERTa, which almost have the same inputs and outputs. Specifically, we add a class `RobertawithAdapter` in `pytorch_transformers/my_modeling_roberta.py`.
- A demo code `[run_example.sh and examples/run_example.py]` about how to use “RobertawithAdapter”, do inference, save model and load model. You can leave the arguments of adapters as default.
- Now it is very easy to use Roberta with adapters. If you only want to use single adapter, for example, fac-adapter, then you can set the argument `meta_fac_adaptermodel='./pretrained_models/fac-adapter/pytorch_model.bin''` and set `meta_lin_adaptermodel=””`. If you want to use both adapters, just set the arguments `meta_fac_adaptermodel` and `meta_lin_adaptermodel` as the path of adapters.
```bash
bash run_example.sh
```
### TODO
- Remove and merge redundant codes
- Support other pre-trained models, such as BERT...

### Contact
Feel free to contact Ruize Wang (rzwang18@fudan.edu.cn) if you have any further questions.

## Pre-trained Adapters

[Google Cloud](https://drive.google.com/drive/folders/12mfLpYq4BTwdbLZnQsdwDJKubM9aOr9f)

## Datasets

[Open Entity](https://www.cs.utexas.edu/~eunsol/html_pages/open_entity.html)
[FIGER](https://drive.google.com/open?id=0B52yRXcdpG6MMnRNV3dTdGdYQ2M)
[TACRED](https://nlp.stanford.edu/projects/tacred/)

### FIGER, OpenEntity and TACRED

[GC](https://drive.google.com/open?id=1HlWw7Q6-dFSm9jNSCh4VaBf1PlGqt9im)
[Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/6ec98dbd931b4da9a7f0/)

### SearchQA, CosmosQA, Quasar-T

[SearchQA + Quasar-T](https://thunlp.oss-cn-qingdao.aliyuncs.com/OpenQA_data.tar.gz)
[SearchQA](https://drive.google.com/drive/u/0/folders/1kBkQGooNyG0h8waaOJpgdGtOnlb1S649)
[CosmosQA](https://github.com/wilburOne/cosmosqa/tree/master/data/)
[Quasar-T](http://curtis.ml.cmu.edu/datasets/quasar/)
[Quasar-T Background Corpus](https://lemurproject.org/clueweb09/)

## New datasets

Entity Typing
[LitWD1K, LitWD19K, LitWD48K](https://zenodo.org/record/4701190#.YW3sbHUza7I)
[WN18RR](https://github.com/villmow/datasets_knowledge_embedding/tree/master/WN18RR)

Relation Classification
[FB15K-237](https://www.microsoft.com/en-us/download/details.aspx?id=52312)
