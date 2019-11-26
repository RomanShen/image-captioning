#!/usr/bin/env bash
id="transformer"
ckpt_path="log_"$id
if [ ! -d $ckpt_path ]; then
  mkdir $ckpt_path
fi
if [ ! -f $ckpt_path"/infos_"$id".pkl" ]; then
start_from=""
else
start_from="--start_from "$ckpt_path
fi

python train.py --id $id --caption_model transformer --noamopt --noamopt_warmup 2000 --label_smoothing 0.0 --input_json /root/PycharmProjects/dataset_rsicd/rsicd.json --input_label_h5 /root/PycharmProjects/dataset_rsicd/data_label.h5 --finetune_cnn_after 16 --input_fc_dir /root/PycharmProjects/dataset_rsicd/data_fc --input_att_dir /root/PycharmProjects/dataset_rsicd/data_att --seq_per_img 5 --batch_size 48 --beam_size 2 --learning_rate 4e-4 --num_layers 6 --input_encoding_size 512 --rnn_size 512 --learning_rate_decay_start 0 --scheduled_sampling_start 0 --checkpoint_path $ckpt_path $start_from --save_checkpoint_every 2000 --language_eval 1 --val_images_use 5000 --max_epochs 1
