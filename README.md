# BioERP
BioERP: a biomedical heterogeneous network-based self-supervised representation learning approach for entity relationship predictions.

# Data description
* DeepViral-Net: a complex heterogeneous networks is downloaded from https://github.com/bio-ontology-research-group/DeepViral/tree/master/data.
* ProGO-Net: a complex heterogeneous networks is downloaded from https://github.com/bio-ontology-research-group/machine-learning-with-ontologies and https://doi.org/10.5281/zenodo.3779900.
* NeoDTI-Net: a complex heterogeneous networks is downloaded from https://github.com/FangpingWan/NeoDTI.git.
* deepDR-Net: a complex heterogeneous networks is downloaded from https://github.com/ChengF-Lab/deepDR.git.
* CTD-DDA, NDFRT-DDA, DrugBank-DDI and STRING-PPI: four single biomedical networks are downloaded from https://github.com/xiangyue9607/BioNEV.git.

# Requirements
BioERP is tested to work under:
* Python 3.6  
* Tensorflow 1.1.4
* tflearn
* numpy 1.14.0
* sklearn 0.19.0

# Quick start
* Download the source code of [BERT](https://github.com/google-research/bert). 
* Manually replace the run_pretraining.py
The network representation model and training regime in BioERP are similar to the original implementation described in "[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://www.aclweb.org/anthology/N19-1423/)". Therefore, the code of network representation of BioERP can be downloaded from https://github.com/google-research/bert. But BERT uses a combination of two tasks, i.e,. masked language learning and the consecutive sentences classification. Nevertheless, different from natural language modeling, meta paths do not have a consecutive relationship. Therefore, BioERP does not involve the continuous sentences training. If you want to run BioERP, please manually replace the run_pretraining.py and run_classifier.py in [BERT](https://github.com/google-research/bert) with these files. 
  
* Download the [BERT-Base, Uncased](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip) model: 12-layer, 768-hidden, 12-heads. 
You can construct a vocab file (vocab.txt) of nodes and modify the config file (bert_config.json) which specifies the hyperparameters of the model.
* Run create_pretraining_data.py to mask metapath sample.  
<pre> python create_pretraining_data.py   \
  --input_file=~path/metapath.txt   \
  --output_file=~path/tf_examples.tfrecord   \
  --vocab_file=~path/uncased_L-12_H-768_A-12/vocab.txt   \ 
  --do_lower_case=True   \  
  --max_seq_length=128   \  
  --max_predictions_per_seq=20   \
  --masked_lm_prob=0.15   \ 
  --random_seed=12345   \
  --dupe_factor=5 </pre>
The max_predictions_per_seq is the maximum number of masked meta path predictions per path sample. masked_lm_prob is the probability for masked token.

* Run run_pretraining.py to train a network representation model based on bio-entity mask mechanism.
<pre> python run_pretraining.py   \  
  --input_file=~path/tf_examples.tfrecord   \  
  --output_dir=~path/Local_RLearing_output   \  
  --do_train=True   \  
  --do_eval=True   \  
  --bert_config_file=~path/uncased_L-12_H-768_A-12/bert_config.json   \ 
  --train_batch_size=32   \  
  --max_seq_length=128   \  
  --max_predictions_per_seq=20   \  
  --num_train_steps=20000   \  
  --num_warmup_steps=10   \  
  --learning_rate=2e-5  </pre>
  
* Run run_classifier.py to train a network representation model based on meta path detection mechanism.
<pre>  python run_classifier.py \
  --task_name=CoLA \
  --do_train=true \
  --do_eval=true \
  --data_dir=~path/all_path \
  --vocab_file=~path/vocab.txt \
  --bert_config_file=~path/bert_config.json \
  --max_seq_length=128 \
  --train_batch_size=256 \
  --learning_rate=2e-5 \
  --num_train_epochs=10 \
  --output_dir=~path/Global_RLearing_output  </pre>

* Run extract_features.py extract_features.py to attain the low-dimensional vectors from two representation models.
<pre> python extract_features.py   \  
  --input_file=~path/node.txt   \  
  --output_file=~path/output.jsonl   \  
  --vocab_file=~path/uncased_L-12_H-768_A-12/vocab.txt   \  
  --bert_config_file=~path/uncased_L-12_H-768_A-12/bert_config.json   \  
  --init_checkpoint=~path/Local_RLearing_output(or Global_RLearing_output)/model.ckpt  \  
  --layers=-1,-2,-3,-4   \  
  --max_seq_length=7   \  
  --batch_size=8   </pre>
  
* Run TDI_NeoDTI.py to predict of the confidence scores between targets and drugs for NeoDTI-Net. 
<pre> python TDI_NeoDTI.py </pre> 

# Please cite our paper if you use this code and data in your work.
@article{BioERP2021, \
title = {BioERP: biomedical heterogeneous network-based self-supervised representation learning approach for entity relationship predictions},  \
author = {Wang Xiaoqi, and Yang Yaning, and Li Kenli, and Li Wentao, and Li Fei, and Peng Shaoliang},  \
journal = {Bioinformatics},  \
year = {2021},  \
doi = {10.1093/bioinformatics/btab565}  \
}

# Contacts
If you have any questions or comments, please feel free to email: xqw@hnu.edu.cn.
