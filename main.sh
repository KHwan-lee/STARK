#! /bin/bash
export LANG=zh_CN.UTF-8


# python main.py -model_name LoraKGE_Layers -ent_r 200 -rel_r 20 -num_ent_layers 2 -num_rel_layers 1 -gpu 0 -dataset ENTITY -learning_rate 3e-1 -using_various_ranks True -predict_result True -batch_size 512
# python main.py -model_name LoraKGE_Layers -ent_r 150 -rel_r 20 -num_ent_layers 2 -num_rel_layers 1 -gpu 0 -dataset WN_CKGE -learning_rate 1e-1 -using_various_ranks True

python data_to_id.py
python cal_features.py
python nodes_sort.py
python main.py -model_name LoraKGE_Layers -ent_r 200 -rel_r 200 -num_ent_layers 20 -num_rel_layers 1 -gpu 0 -dataset FB_CKGE -learning_rate 1e-1 -using_various_ranks True -patience 10
python main.py -model_name LoraKGE_Layers -ent_r 200 -rel_r 200 -num_ent_layers 20 -num_rel_layers 1 -gpu 0 -dataset WN_CKGE -learning_rate 2e-1 -using_various_ranks True -patience 10
python main.py -model_name LoraKGE_Layers -ent_r 200 -rel_r 200 -num_ent_layers 20 -num_rel_layers 1 -gpu 0 -dataset FACT -learning_rate 1e-1 -using_various_ranks True -patience 10
python main.py -model_name LoraKGE_Layers -ent_r 200 -rel_r 150 -num_ent_layers 20 -num_rel_layers 8 -gpu 0 -dataset HYBRID -learning_rate 1e-1 -using_various_ranks True -patience 10
python main.py -model_name LoraKGE_Layers -ent_r 200 -rel_r 200 -num_ent_layers 20 -num_rel_layers 1 -gpu 0 -dataset ENTITY -learning_rate 1e-1 -using_various_ranks True -patience 10
python main.py -model_name LoraKGE_Layers -ent_r 200 -rel_r 200 -num_ent_layers 20 -num_rel_layers 10 -gpu 0 -dataset RELATION -learning_rate 1e-1 -using_various_ranks True -patience 10