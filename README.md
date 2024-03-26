# LayoutDIT
LayoutDIT is a layout-aware end-to-end document image translation (DIT) framework. It effectively incorporates the layout information into DIT in an end-to-end way and significantly improves the translation for document images of diverse domains and layouts/formats in our experiments.

Our paper [LayoutDIT: Layout-Aware End-to-End Document Image Translation with Multi-Step Conductive Decoder](https://aclanthology.org/2023.findings-emnlp.673/) has been accepted by EMNLP 2023.

DITrans is a new benchmark dataset for document image translation built with three document domains and fine-grained human-annotated labels, enabling the research on En-Zh DIT, reading order detection, and layout analysis. Note that the DITrans dataset can only be used for non-commercial research purposes. For scholars or organizations who want to use the dataset, please send an application via email to us (zhangzhiyang2020@ia.ac.cn). When submitting the application to us, please list or attach 1-2 of your publications in the recent 2 years to indicate that you (or your team) do research in the related research fields of document image processing or machine translation. At present, this dataset is only freely available to scholars in the above-mentioned fields. We will give you the download links and decompression passwords for the dataset after your letter has been received and approved.

# Dependency
```python
torch==1.8.1
transformers==4.30.0
jieba==0.42.1
nltk==3.8.1
```

# Training
```python
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4


python -m torch.distributed.launch --nproc_per_node=2 --master_port 29930 train.py \
    --train_folder /path/to/trainset_folder \
    --base_model_type layoutlm \
    --model_name layoutlm-base-uncased \
    --output_dir /path/to/ckpts_dir \
    --log_dir /path/to/logs_dir \
    --cache_dir /path/to/cache_dir \
    --max_source_length 512 \
    --max_target_length 512 \
    --num_hidden_layers 6 \
    --cached_train_features_file /path/to/pre-processed_train_feat_pt_file \
    --do_lower_case \
    --per_gpu_train_batch_size 16 \
    --learning_rate 7e-5 \
    --label_smoothing 0.1 \
    --num_training_steps 200000 \
    --num_warmup_steps 10000 \
    --random_prob 0.80 \
    --keep_prob 0.05 \
    --logging_steps 2 \
    --save_steps $[200000 / 40] \
    --fp16 \
    --fp16_opt_level O1 \
    --senseg_encoder_num_hidden_layers 1 \
    --senseg_task_loss_relative_weight 1 \
    --src_sen_max_len 128 \
    --tgt_sen_max_len 128 \
    --tgt_max_position_embeddings 256 \
    --trans_decoder_num_hidden_layers 6 \
    --trans_task_relative_weight 1 \
    --trans_decoder_max_fwd_tokens 768
```

# Decoding
```python
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

python decode.py \
    --test_folder /path/to/testset_folder \
    --cached_test_features_dir /path/to/pre-processed_test_feat_pt_file_dir \
    --cached_test_features_filename test_feat_pt_filename \
    --test_output_dir /path/to/decoding_results_dir \
    --model_recovery_dir /path/to/used_ckpt_dir \
    --cache_dir /path/to/cache_dir \
    --do_lower_case \
    --batch_size 32 \
    --fp16 \
    --src_sen_max_len 128 \
    --tgt_sen_max_len 128 \
    --trans_decoder_max_fwd_tokens 8960 \
    --num_beams 4 \
    --early_stopping 
```

# Results
Our model achieves 46.97 on DITrans-Report En-Zh DIT task. Detailed results are reported as follow.
* Multi-domain learning: In this setting, models are trained on DITrans' all three domains and tested on each domain.
  | Method | End-to-End | # Params (M) | DITrans-Report | DITrans-News | DITrans-Advertisement |
  | --- | --- | --- | --- | --- | --- |
  | DocHandler-1 | No | 142 | 21.47 | 26.28 | 23.14 |
  | DocHandler-2 | No | 172 | 24.80 | 27.34 | 24.99 |
  | MGTrans-DETR | No | 212 | 28.30 | 27.61 | 25.76 |
  | MGTrans-Conv | No | 238 | 28.99 | 28.84 | 25.63 |
  | LayoutLM-Dec | Yes | 232 | 45.86 | 40.66 | 36.90 |
  | LiLT-Dec | Yes | 250 | 46.11 | 40.37 | 36.22 |
  | LayoutDIT-Cascade | No | 293 | 36.78 | 34.56 | 35.36 |
  | LayoutDIT | Yes | 206 | 46.97 | 38.82 | 45.72 |
* Zero-shot cross-domain transfer: In this setting, models are trained only on DITrans-Report domain and are tested on each domain. This setting evaluates the cross-domain ability of models.
  | Method | End-to-End | # Params (M) | DITrans-Report | DITrans-News | DITrans-Advertisement |
  | --- | --- | --- | --- | --- | --- |
  | DocHandler-1 | No | 142 | 21.80 | 16.87 | 13.55 |
  | DocHandler-2 | No | 172 | 25.19 | 16.88 | 13.60 |
  | MGTrans-DETR | No | 212 | 27.34 | 16.26 | 13.09 |
  | MGTrans-Conv | No | 238 | 29.51 | 17.78 | 14.07 |
  | LayoutLM-Dec | Yes | 232 | 45.71 | 27.56 | 21.40 |
  | LiLT-Dec | Yes | 250 | 46.61 | 23.71 | 18.14 |
  | LayoutDIT-Cascade | No | 293 | 39.85 | 21.89 | 16.62 |
  | LayoutDIT | Yes | 206 | 46.70 | 29.01 | 29.46 |
* More experiment results including low-resource DIT, En-De/Fr results, and ablations can be found in our paper.
