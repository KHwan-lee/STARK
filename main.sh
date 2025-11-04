#! /bin/bash
export LANG=zh_CN.UTF-8



python data_to_id.py
python cal_features.py
python nodes_sort.py


python main.py -model_name LoraKGE_Layers -ent_r 200 -rel_r 200 -num_ent_layers 20 -num_rel_layers 1 -gpu 0 -learning_rate 1e-1 -using_various_ranks True -neg_ratio 10 -patience 10 -margin 8.0 -lambda_dyn 1.0 -fusion_weight 0.2 -hub_scale 0.05 -dataset FB_CKGE





