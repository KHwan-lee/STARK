#! /bin/bash
export LANG=zh_CN.UTF-8


python main.py -model_name LoraKGE_Layers -ent_r 200 -rel_r 20 -num_ent_layers 2 -num_rel_layers 1 -gpu 0 -dataset ENTITY -learning_rate 3e-1 -using_various_ranks True -predict_result True -batch_size 512
#python main.py -model_name LoraKGE_Layers -ent_r 150 -rel_r 20 -num_ent_layers 2 -num_rel_layers 1 -gpu 0 -dataset WN_CKGE -learning_rate 1e-1 -using_various_ranks True