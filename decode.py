
import jieba
import time
import argparse
import os
import json
import math
import tqdm
from nltk.translate.bleu_score import sentence_bleu

import torch
from torch.utils.data import (DataLoader, SequentialSampler)

from transformers import BertConfig, BertTokenizer

from modeling import utils
from modeling.utils import wdp_detokenizer, spm_detokenizer
from modeling.utils import set_seed
from modeling.utils import senesg_convert_tgt_layout_inputs_to_tokens_wi_eos, trans_convert_src_tgt_inputs_to_tokens
from modeling.config import LayoutlmConfig, BertForSeq2SeqConfig
from modeling.modeling import LayoutlmReorderingSensegTransIncr
from modeling.dataset import load_and_cache_layoutlm_examples
from modeling.utils import convert_src_layout_inputs_to_tokens, convert_tgt_layout_inputs_to_tokens
from modeling import dataset
from modeling import predefined_constant
from modeling.predefined_constant import senseg_task_ctg_to_id_map, senseg_task_id_to_ctg_map
from modeling.utils import Chinese_tokenizer as tgt_tokenizer


import logging
logger = logging.getLogger(__name__)

# other external imports.
import jieba
from nltk.translate.bleu_score import sentence_bleu
from sacrebleu.metrics import CHRF
chrf = CHRF(word_order=2) # word_order=2 to be chrf++.

MODEL_CLASSES = {
    'bert': (BertConfig, BertTokenizer),
    'layoutlm': (LayoutlmConfig, BertTokenizer),
}


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_file", type=str, help="Input file")
    parser.add_argument("--input_folder", type=str, help="Input folder")
    parser.add_argument('--num_subset_instances', type=int, default=0,
                        help="Decode a subset of the input dataset.")
    parser.add_argument("--base_model_type", default=None, type=str, required=True, help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--layoutlm_only_layout", action='store_true')
    parser.add_argument("--model_name", default=None, type=str, required=True, help="Path to pre-trained model or shortcut name.")
    parser.add_argument("--model_dir", default=None, type=str, required=True,
                        help="dir to the model checkpoint.")
    parser.add_argument("--output_file", type=str, help="output json file to store the decoding results.")
    parser.add_argument("--metric_file", type=str, help="txt file to store the metric.")
    parser.add_argument("--log_dir", default=None, type=str,
                        help="The output directory where the log will be written.")
    parser.add_argument("--config_path", default=None, type=str,
                        help="Path to config.json for the model.")
    parser.add_argument("--tokenizer_name", default=None, type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default=None, type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_source_length", default=None, type=int,
                        help="The maximum total source sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--max_target_length", default=None, type=int,
                        help="The maximum total target sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--cached_feature_file", type=str, default=None)
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument('--batch_size', type=int, default=4,
                        help="Batch size for decoding.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    
    # for trans task.
    parser.add_argument("--src_sen_max_len", default=128, type=int,
                        help="trans task max src sentence length.")
    parser.add_argument("--tgt_sen_max_len", default=128, type=int,
                        help="trans task max tgt sentence length.")
    parser.add_argument("--trans_decoder_max_fwd_tokens", default=1792, type=int,
                        help="The max tokens (src lang seq tokens + tgt lang seq tokens) for the trans decoder to do a single forward pass. Setting this threshold to prevent the possible OOM error.")
    parser.add_argument("--num_beams", default=4, type=int,
                        help="trans task decoding beam size.")
    parser.add_argument("--length_penalty", default=1.0, type=float,
                        help="beam search length penalty.")
    parser.add_argument("--early_stopping", action='store_true',
                        help="beam search early stopping action.")
    parser.add_argument("--max_generation_len_ratio", default=1.2, type=float,
                        help="the max length of generated tgt seq relative to src seq.")
    parser.add_argument("--max_generation_abs_len", default=128, type=int,
                        help="the max absolute generation length of tgt seq.")
    
    args = parser.parse_args()
    return args


def prepare(args):
    os.makedirs(args.log_dir, exist_ok=True)
    json.dump(args.__dict__, open(os.path.join(
        args.log_dir, 'decoding_args.json'), 'w'), indent=2)
    
    # setup device.
    args.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()

    # args.device = torch.device("cpu")
    # args.n_gpu = 1

    # Setup logging
    log_filepath = f"{args.log_dir}/decode.{{{time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime())}}}.log"
    logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO ,
                        filename=log_filepath,
                        filemode="a")
    
    # set random seed.
    set_seed(args.seed, args.n_gpu)

    logger.info("Training/evaluation args %s", args)

    if args.fp16:
        try:
            import apex
            apex.amp.register_half_function(torch, 'einsum')
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        
    # Add the required args for senseg task.
    args.senseg_task_ctg_to_id_map = senseg_task_ctg_to_id_map
    args.senseg_task_id_to_ctg_map = senseg_task_id_to_ctg_map

    # Add the required args for trans task.
    args.tgt_vocab_size = tgt_tokenizer.vocab_size


def get_model_and_tokenizer(args):
    # prepare model config instance.
    config_file = args.config_path if args.config_path else os.path.join(args.model_dir, "config.json")
    logger.info("Read decoding config from: %s" % config_file)

    config_class, tokenizer_class = MODEL_CLASSES[args.base_model_type]
    model_config = config_class.from_json_file(
        config_file
    )
    model_config.trans_decoder_max_fwd_tokens = args.trans_decoder_max_fwd_tokens
    
    # prepare tokenizer.
    if args.base_model_type == 'layoutlm':
        if args.tokenizer_name is not None:
            tokenizer_name = args.tokenizer_name
        else:
            tokenizer_name = 'bert' + args.model_name[8:]
        tokenizer = tokenizer_class.from_pretrained(
            tokenizer_name, do_lower_case=args.do_lower_case, cache_dir=args.cache_dir if args.cache_dir else None)
    else:
        tokenizer = tokenizer_class.from_pretrained(
            args.tokenizer_name if args.tokenizer_name else args.model_name,
            do_lower_case=args.do_lower_case, cache_dir=args.cache_dir if args.cache_dir else None)
        
    # prepare model sturcture and load the ckpt weights.
    model_dir = args.model_dir
    args.model_path = os.path.join(model_dir, "pytorch_model.bin")
    assert os.path.exists(model_dir), 'model_dir ' + model_dir + ' not exists!'
    logger.info("***** Recover model: %s *****", model_dir)
    # model = LayoutlmEncoderDecoder.from_pretrained(
    #     pretrained_model_name_or_path=args.model_path, base_model_type=args.base_model_type, config=model_config,) 

    # Incremental decoding to save time and memory to a large extent.
    model = LayoutlmReorderingSensegTransIncr.from_pretrained(
        pretrained_model_name_or_path=args.model_path, base_model_type=args.base_model_type, config=model_config)

    if args.fp16: 
        model.half()

    torch.cuda.empty_cache()

    model.to(args.device)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    return model, tokenizer

def decode(args, model, tokenizer, tgt_tokenizer):

    # prepare testset raw features.
    example_path = args.input_file if args.input_file else args.input_folder
    to_pred = load_and_cache_layoutlm_examples(
            example_path, tokenizer, tgt_tokenizer, local_rank=-1,
            cached_features_file=args.cached_feature_file, shuffle=False, layout_flag=args.base_model_type == 'layoutlm',
            max_src_length=args.max_source_length,
        )
    
    if args.num_subset_instances > 0:
        logger.info(f"Decoding the first {args.num_subset_instances} instances of the full testset.")
        to_pred = to_pred[:args.num_subset_instances]
    
    # reordering the instance order for better batching and decoding speed-up.
    for i, item in enumerate(to_pred):
        item["instance_id"] = i
    to_pred = sorted(to_pred, key=lambda item: len(item["source_ids"]), reverse=False)
    sorted_instance_ids = [item["instance_id"] for item in to_pred]


    # memorize the input_lines, target_lines and target_geo_scores for subsequent instance-level bleu calculation.
    # for reordering task.
    input_lines = convert_src_layout_inputs_to_tokens(to_pred, tokenizer.convert_ids_to_tokens, args.max_source_length-1,
                                                          layout_flag=args.base_model_type == 'layoutlm')
    target_lines = convert_tgt_layout_inputs_to_tokens(to_pred, tokenizer.convert_ids_to_tokens, args.max_target_length-1,
                                                           layout_flag=args.base_model_type == 'layoutlm')
    # for senseg task.
    target_lines_wi_eos = senesg_convert_tgt_layout_inputs_to_tokens_wi_eos(to_pred, tokenizer.convert_ids_to_tokens, args.max_target_length-1, senseg_task_id_to_ctg_map=args.senseg_task_id_to_ctg_map, bos_tag=predefined_constant.BOS_TAG, ios_tag=predefined_constant.IOS_TAG, eos_token=predefined_constant.RAW_DATASET_EOS_TOKEN,
                                                           layout_flag=args.base_model_type == 'layoutlm')
    
    # for trans task.
    src_list_sen_tokens_list, tgt_list_sen_tokens_list = trans_convert_src_tgt_inputs_to_tokens(to_pred, tokenizer.convert_ids_to_tokens, tgt_tokenizer.convert_ids_to_tokens, args.src_sen_max_len, args.tgt_sen_max_len-1)

    target_geo_scores = [x['bleu'] for x in to_pred]


    # prepare test dataset and dataloader based on the testset raw features.
    to_pred_dataset = dataset.Seq2seqDatasetForLayoutlm(
        features=to_pred, max_source_len=args.max_source_length,
        max_target_len=args.max_target_length, vocab_size=tokenizer.vocab_size,
        cls_id=tokenizer.cls_token_id, sep_id=tokenizer.sep_token_id, pad_id=tokenizer.pad_token_id, mask_id=tokenizer.mask_token_id, random_prob=0.0, keep_prob=1.0,
        offset=0, num_training_instances=len(to_pred),
        layout_flag=args.base_model_type == 'layoutlm',
        cls_index=predefined_constant.CLS_TOKEN_INDEX,
        pad_index=predefined_constant.PAD_TOKEN_INDEX,
        cls_layout=predefined_constant.CLS_TOKEN_LAYOUT,
        pad_layout=predefined_constant.PAD_TOKEN_LAYOUT,
        sep_layout=predefined_constant.SEP_TOKEN_LAYOUT,
        mask_layout=predefined_constant.MASK_TOKEN_LAYOUT,
        senseg_label_pad_id=predefined_constant.SENSEG_LABEL_PAD_ID,
        src_sen_max_len=args.src_sen_max_len,
        tgt_sen_max_len=args.tgt_sen_max_len,
        target_sen_pad_id=tgt_tokenizer.pad_token_id,
        target_sen_sep_id=tgt_tokenizer.sep_token_id,
        target_sen_eos_id=tgt_tokenizer.cls_token_id,
    )

    to_pred_sampler = SequentialSampler(to_pred_dataset)
    data_collator = utils.BatchListToBatchTensorsCollator(
        max_source_len=args.max_source_length,
        max_target_len=args.max_target_length,
    )
    to_pred_dataloader = DataLoader(
            to_pred_dataset, sampler=to_pred_sampler,
            shuffle=False,
            batch_size=args.batch_size,
            collate_fn=data_collator,
            num_workers=0)

    to_pred_iterator = tqdm.tqdm(
            to_pred_dataloader, initial=len(to_pred_dataloader),)
    
    # prepare the mode of model for decoding.
    model.zero_grad()
    model.eval()

    # prepare the file to log decoding results.
    fout = open(args.output_file, "w")

    # start decoding.
    global_decoding_step = 0
    tokenizer_special_token_ids = tokenizer.convert_tokens_to_ids([tokenizer.sep_token, tokenizer.pad_token, tokenizer.cls_token, tokenizer.mask_token])
    tgt_tokenizer_special_token_ids = tgt_tokenizer.convert_tokens_to_ids([tgt_tokenizer.sep_token, tgt_tokenizer.pad_token, tgt_tokenizer.cls_token, tgt_tokenizer.mask_token, tgt_tokenizer.eos_token])
    for step, batch in enumerate(to_pred_iterator):
            batch = tuple(t.to(args.device) for t in batch)

            # ===================================== to do ================================
            inputs = {'source_idxys': batch[0],
                      'num_effective_source_tokens': batch[7],
                      'tokenizer': tokenizer,
                      'tgt_tokenizer': tgt_tokenizer,
                      'num_beams': args.num_beams,
                      'length_penalty': args.length_penalty,
                      'early_stopping': args.early_stopping,
                      'max_generation_len_ratio': args.max_generation_len_ratio,
                      'max_generation_abs_len': args.max_generation_abs_len,
                      }
            num_effective_source_tokens_list = inputs["num_effective_source_tokens"].tolist()

            
            with torch.no_grad():
                model_res = model.greedy_decode_beam_search(**inputs)
                reordering_task_decoded_token_ids = model_res["reordering_task_res"] # shape = [batch_size, max_src_seq_len_cur_batch+1].
                senseg_task_predicted_label_ids = model_res["senseg_task_res"] # shape = [batch_size, max_src_seq_len_cur_batch+1].
                trans_task_decoded_token_ids_tuple = model_res["trans_task_res"] # each element of this tuple corresponds to a instance, and of the shape = [num_sen_this_instance, max_generation_len].
                
            
            # ===================================== to do ================================

            reordering_task_decoded_token_ids_list = reordering_task_decoded_token_ids.tolist()
            senseg_task_predicted_label_ids_list = senseg_task_predicted_label_ids.tolist()
            trans_task_decoded_token_ids_list_list = [item.tolist() for item in trans_task_decoded_token_ids_tuple]
            for i in range(len(reordering_task_decoded_token_ids_list)):
                reordering_task_decoded_token_ids = reordering_task_decoded_token_ids_list[i]
                senseg_task_predicted_label_ids = senseg_task_predicted_label_ids_list[i]
                trans_task_decoded_token_ids_list = trans_task_decoded_token_ids_list_list[i]


                # num_effective_source_tokens_list = inputs["num_effective_source_tokens"].tolist()
                num_effective_source_tokens = num_effective_source_tokens_list[i]
                effective_decoded_token_ids = reordering_task_decoded_token_ids[:num_effective_source_tokens]
                effective_predicted_label_ids = senseg_task_predicted_label_ids[:num_effective_source_tokens]


                # decoded_token_ids -> decoded_seq_wo_eos/decoded_seq_wi_eos.
                effective_decoded_token_ids_wo_eos = []
                effective_decoded_token_ids_wi_eos = []

                tok_num = 0
                while tok_num < len(effective_decoded_token_ids) - 1:
                    
                    token_id = effective_decoded_token_ids[tok_num]
                    label_id = effective_predicted_label_ids[tok_num]

                    if token_id in tokenizer_special_token_ids:
                        tok_num += 1
                        continue
                
                    # for decoded seq wo eos token.
                    effective_decoded_token_ids_wo_eos.append(token_id)

                    # for decoded seq wi eos token.
                    is_next_token_bos_tag = (effective_predicted_label_ids[tok_num+1] == predefined_constant.BOS_TAG_ID)
                    if not is_next_token_bos_tag:
                        effective_decoded_token_ids_wi_eos.append(token_id)
                    else:
                        effective_decoded_token_ids_wi_eos.append(token_id)
                        effective_decoded_token_ids_wi_eos.append(tokenizer.convert_tokens_to_ids(predefined_constant.RAW_DATASET_EOS_TOKEN))

                    tok_num += 1
                
                # Append the last token and eos token.
                effective_decoded_token_ids_wo_eos.append(effective_decoded_token_ids[tok_num])
                effective_decoded_token_ids_wi_eos.append(effective_decoded_token_ids[tok_num])
                effective_decoded_token_ids_wi_eos.append(tokenizer.convert_tokens_to_ids(predefined_constant.RAW_DATASET_EOS_TOKEN))

                effective_trans_task_decoded_token_ids_list = []
                for token_ids in trans_task_decoded_token_ids_list:
                    token_ids_ = [token_id for token_id in token_ids if token_id not in tgt_tokenizer_special_token_ids]
                    effective_trans_task_decoded_token_ids_list.append(token_ids_)


                decoded_seq_wo_eos = tokenizer.convert_ids_to_tokens(effective_decoded_token_ids_wo_eos)
                decoded_seq_wo_eos = ' '.join(wdp_detokenizer(decoded_seq_wo_eos))
                decoded_seq_wi_eos = tokenizer.convert_ids_to_tokens(effective_decoded_token_ids_wi_eos)
                decoded_seq_wi_eos = ' '.join(wdp_detokenizer(decoded_seq_wi_eos))


                if '\n' in decoded_seq_wo_eos:
                    decoded_seq_wo_eos = " [X_SEP] ".join(decoded_seq_wo_eos.split('\n'))
                if '\n' in decoded_seq_wi_eos:
                    decoded_seq_wi_eos = " [X_SEP] ".join(decoded_seq_wi_eos.split('\n'))


                trans_task_decoded_seq_list = []
                for effective_trans_task_decoded_token_ids in effective_trans_task_decoded_token_ids_list:
                    decoded_seq = tgt_tokenizer.convert_ids_to_tokens(effective_trans_task_decoded_token_ids)
                    decoded_seq = spm_detokenizer(decoded_seq)
                    if '\n' in decoded_seq:
                        decoded_seq = "[X_SEP]".join(decoded_seq.split('\n'))
                    trans_task_decoded_seq_list.append(decoded_seq)
                trans_task_decoded_seq = ''.join(trans_task_decoded_seq_list)

                trans_task_decoded_seq_seg = ""
                for item in trans_task_decoded_seq_list:
                    trans_task_decoded_seq_seg += f"{' '.join(jieba.cut(item))} {predefined_constant.RAW_DATASET_EOS_TOKEN} "
                trans_task_decoded_seq_seg = trans_task_decoded_seq_seg[:-1]

                # retrive the target_seq.
                target_seq_wo_eos = target_lines[global_decoding_step]
                target_seq_wo_eos = " ".join(wdp_detokenizer(target_seq_wo_eos))
                target_seq_wi_eos = target_lines_wi_eos[global_decoding_step]
                target_seq_wi_eos = " ".join(wdp_detokenizer(target_seq_wi_eos))
                trans_task_target_seq_list = []
                for sen_tokens_list in tgt_list_sen_tokens_list[global_decoding_step]:
                    target_seq = ''.join(spm_detokenizer(sen_tokens_list))
                    trans_task_target_seq_list.append(target_seq)
                trans_task_target_seq = ''.join(trans_task_target_seq_list)


                trans_task_target_seq_seg = ""
                for item in trans_task_target_seq_list:
                    trans_task_target_seq_seg += f"{' '.join(jieba.cut(item))} {predefined_constant.RAW_DATASET_EOS_TOKEN} "
                trans_task_target_seq_seg = trans_task_target_seq_seg[:-1]
                

                # calculate instance-level bleu.
                instance_bleu_wo_eos = sentence_bleu([target_seq_wo_eos.split()], decoded_seq_wo_eos.split())
                instance_bleu_wi_eos = sentence_bleu([target_seq_wi_eos.split()], decoded_seq_wi_eos.split())
                instance_bleu_trans = sentence_bleu([list(jieba.cut(trans_task_target_seq))], list(jieba.cut(trans_task_decoded_seq)))

                # calculate instance-level chrf++ for trans task.
                instance_chrf_score = chrf.sentence_score(
                    hypothesis=" ".join(jieba.cut(trans_task_decoded_seq)),
                    references=[" ".join(jieba.cut(trans_task_target_seq))],
                ).score

                # write decoding res to json file.
                instance_id = sorted_instance_ids[global_decoding_step]
                geo_score = target_geo_scores[global_decoding_step]
                # reordering task.
                reordering_task_dict = {
                    "bleu": instance_bleu_wo_eos,
                    "model_decoding_res": decoded_seq_wo_eos,
                    "target": target_seq_wo_eos,
                }
                # senseg task.
                senseg_task_dict = {
                    "bleu": instance_bleu_wi_eos,
                    "model_prediction_res": decoded_seq_wi_eos,
                    "target": target_seq_wi_eos,
                }
                # trans task.
                trans_task_dict = {
                    "bleu": instance_bleu_trans,
                    "chrf++": instance_chrf_score,
                    "model_hypothesis": " ".join(jieba.cut(trans_task_decoded_seq)),
                    "target": " ".join(jieba.cut(trans_task_target_seq)),
                    "model_hypothesis_senseg": trans_task_decoded_seq_seg,
                    "target_senseg": trans_task_target_seq_seg,
                }

                item_dict = {
                    "instance_id": instance_id,
                    "geo_score": geo_score,
                    "reordering_task_res": reordering_task_dict,
                    "senseg_task_res": senseg_task_dict,
                    "trans_task_res": trans_task_dict,
                }

                fout.write(f"{json.dumps(item_dict, ensure_ascii=False)}\n")
                # fout.write('{}\t{:.8f}\t{:.8f}\t{:.8f}\t{}\t{}\t{}\t{}\n'.format(instance_id, instance_bleu_wi_eos, instance_bleu_wo_eos, geo_score, decoded_seq_wi_eos, target_seq_wi_eos, decoded_seq_wo_eos, target_seq_wo_eos))

                global_decoding_step += 1

    fout.close()

    # log the averaged instance-level bleu.
    metric_file = open(args.metric_file, "w")
    fout = open(args.output_file, "r", encoding="utf8")
    bleu_score_wo_eos = bleu_score_wi_eos = geo_score = bleu_score_trans = chrf_score_trans = {}
    total_bleu_wo_eos = total_bleu_wi_eos = total_geo = total_bleu_trans = total_chrf_trans = 0.0
    for line in fout.readlines():
        line = line.strip()
        line_dict = json.loads(line)

        instance_id = line_dict["instance_id"]
        instance_geo_score = line_dict["geo_score"]
        instance_bleu_wo_eos = line_dict["reordering_task_res"]["bleu"]
        instance_bleu_wi_eos = line_dict["senseg_task_res"]["bleu"]
        instance_bleu_trans = line_dict["trans_task_res"]["bleu"]
        instance_chrf_trans = line_dict["trans_task_res"]["chrf++"]

        bleu_score_wo_eos[int(instance_id)] = float(instance_bleu_wo_eos)
        total_bleu_wo_eos += float(instance_bleu_wo_eos)
        bleu_score_wi_eos[int(instance_id)] = float(instance_bleu_wi_eos)
        total_bleu_wi_eos += float(instance_bleu_wi_eos)
        geo_score[int(instance_id)] = float(instance_geo_score)
        total_geo += float(instance_geo_score)
        bleu_score_trans[int(instance_id)] = float(instance_bleu_trans)
        total_bleu_trans += float(instance_bleu_trans)
        chrf_score_trans[int(instance_id)] = float(instance_chrf_trans)
        total_chrf_trans += float(instance_chrf_trans)

    avg_bleu_wo_eos_info = f"avg instance-level bleu wo eos token: {round(100 * total_bleu_wo_eos / len(bleu_score_wo_eos), 2)}"
    avg_bleu_wi_eos_info = f"avg instance-level bleu wi eos token: {round(100 * total_bleu_wi_eos / len(bleu_score_wi_eos), 2)}"
    avg_geo_info = f"avg instance-level geo: {round(100 * total_geo / len(geo_score), 2)}"
    avg_bleu_trans_info = f"avg instance-level bleu trans task: {round(100 * total_bleu_trans / len(bleu_score_trans), 2)}"
    avg_chrf_trans_info = f"avg instance-level chrf++ trans task: {round(total_chrf_trans / len(chrf_score_trans), 2)}"
    
    logger.info(avg_bleu_wo_eos_info)
    logger.info(avg_bleu_wi_eos_info)
    logger.info(avg_geo_info)
    logger.info(avg_bleu_trans_info)
    logger.info(avg_chrf_trans_info)
    
    metric_file.write(f"{avg_bleu_wo_eos_info}\n")
    metric_file.write(f"{avg_bleu_wi_eos_info}\n")
    metric_file.write(f"{avg_geo_info}\n")
    metric_file.write(f"{avg_bleu_trans_info}\n")
    metric_file.write(f"{avg_chrf_trans_info}\n")

    metric_file.close()
    fout.close()


def main():
    args = get_args()
    prepare(args)

    model, tokenizer = get_model_and_tokenizer(args)

    decode(args, model, tokenizer, tgt_tokenizer)

if __name__ == "__main__":
    main()

    