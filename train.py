import torch
from torch.utils.data import (DataLoader, SequentialSampler)
from torch.utils.data.distributed import DistributedSampler

from tensorboardX import SummaryWriter

from transformers import BertConfig, BertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup, AutoConfig

from modeling import dataset
from modeling import utils
from modeling.utils import set_seed
from modeling import predefined_constant
from modeling.config import LayoutlmConfig, BertForSeq2SeqConfig
from modeling.modeling import LayoutlmReorderingSensegTrans
from modeling.predefined_constant import senseg_task_ctg_to_id_map, senseg_task_id_to_ctg_map
from modeling.utils import Chinese_tokenizer as tgt_tokenizer

import time
import glob
import os
import json
import argparse
import tqdm
import logging
logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'bert': (BertConfig, BertTokenizer),
    'layoutlm': (LayoutlmConfig, BertTokenizer),
}

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_file", default=None, type=str,
                        help="Training data (json format) for training. Keys: source and target")
    parser.add_argument("--train_folder", default=None, type=str,
                        help="Training data folder for training. Keys: source and target")
    parser.add_argument("--base_model_type", default=None, type=str, required=True, help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--layoutlm_only_layout", action='store_true')
    parser.add_argument("--model_name", default=None, type=str, required=True, help="Path to pre-trained model or shortcut name.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints and predictions will be written.")
    parser.add_argument("--model_recovery_dir", default=None, type=str,
                        help="The ckpt directory where the model will load for test/finetune.")
    parser.add_argument("--log_dir", default=None, type=str,
                        help="The output directory where the log will be written.")
    parser.add_argument("--config_name", default=None, type=str,
                        help="Pretrained config name or path if none the same as model_name")
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
    parser.add_argument("--num_hidden_layers", default=8, type=int,
                        help="reordering task encoder decoder num layers.")
    parser.add_argument("--cached_train_features_file", default=None, type=str,
                        help="Cached training features file")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--per_gpu_train_batch_size", default=None, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--learning_rate", default=7e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--label_smoothing", default=0.1, type=float,
                        help="Label smoothing.")
    parser.add_argument("--num_training_steps", default=-1, type=int,
                        help="set total number of training steps to perform")
    parser.add_argument("--num_training_epochs", default=None, type=int,
                        help="set total number of training epochs to perform (--num_training_steps has higher priority)")
    parser.add_argument("--num_warmup_steps", default=500, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--random_prob", default=0.05, type=float,
                        help="prob to random replace a masked token")
    parser.add_argument("--keep_prob", default=0.80, type=float,
                        help="prob to keep no change for a masked token")
    parser.add_argument('--logging_steps', type=int, default=20,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=4000,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    
    # For senseg task.
    parser.add_argument("--senseg_encoder_num_hidden_layers", default=2, type=int,
                        help="senseg task encoder num layers.")
    parser.add_argument("--senseg_task_loss_relative_weight", default=1.0, type=float,
                        help="the relative loss weight of senseg task compared to word reordering task.")
    
    # for trans task.
    parser.add_argument("--src_sen_max_len", default=128, type=int,
                        help="trans task max src sentence length.")
    parser.add_argument("--tgt_sen_max_len", default=128, type=int,
                        help="trans task max tgt sentence length.")
    parser.add_argument("--tgt_max_position_embeddings", default=128, type=int,
                        help="trans task max src sentence length.")
    parser.add_argument("--trans_decoder_num_hidden_layers", default=4, type=int,
                        help="trans task decoder num layers.")
    parser.add_argument("--trans_task_relative_weight", default=1, type=int,
                        help="the relative loss weight of trans task compared to word reordering task.")
    parser.add_argument("--trans_decoder_max_fwd_tokens", default=16384, type=int,
                        help="The max tokens (src lang seq tokens + tgt lang seq tokens) for the trans decoder to do a single forward pass. Setting this threshold to prevent the possible OOM error.")

    args = parser.parse_args()
    return args


def prepare(args):

    os.makedirs(args.output_dir, exist_ok=True)
    json.dump(args.__dict__, open(os.path.join(
        args.output_dir, 'train_args.json'), 'w'), indent=2)

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda: 
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank) # If I use device = torch.device("cuda:1"), I always got RuntimeError: CUDA error: an illegal memory access was encountered error. But when I set a specific gpu by torch.cuda.set_device(1), everything is fine.

        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1

    args.device = device


    # Setup logging
    log_filepath = f"{args.log_dir}/train.{{{time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime())}}}.log"

    logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
                        filename=log_filepath,
                        filemode="a")
    
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args.seed, args.n_gpu)

    logger.info("Training/evaluation args %s", args)

    # Before we do anything with models, we want to ensure that we get fp16 execution of torch.einsum if args.fp16 is set.
    # Otherwise it'll default to "promote" mode, and we'll get fp32 operations. Note that running `--fp16_opt_level="O2"` will
    # remove the need for this code, but it is still valid.
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
    config_class, tokenizer_class = MODEL_CLASSES[args.base_model_type]
    base_model_config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name,
        cache_dir=args.cache_dir if args.cache_dir else None)
    config = BertForSeq2SeqConfig.from_exist_config(
        config=base_model_config, 
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
        label_smoothing=args.label_smoothing,
        base_model_type=args.base_model_type,
        num_hidden_layers=args.num_hidden_layers,
        layoutlm_only_layout=args.layoutlm_only_layout,
        senseg_encoder_num_hidden_layers=args.senseg_encoder_num_hidden_layers,
        senseg_task_loss_relative_weight=args.senseg_task_loss_relative_weight,
        senseg_task_ctg_to_id_map=args.senseg_task_ctg_to_id_map,
        senseg_task_id_to_ctg_map=args.senseg_task_id_to_ctg_map,
        src_sen_max_len=args.src_sen_max_len,
        tgt_sen_max_len=args.tgt_sen_max_len,
        tgt_vocab_size=args.tgt_vocab_size,
        tgt_max_position_embeddings=args.tgt_max_position_embeddings,
        trans_decoder_num_hidden_layers=args.trans_decoder_num_hidden_layers,
        trans_task_relative_weight=args.trans_task_relative_weight,
        trans_decoder_max_fwd_tokens=args.trans_decoder_max_fwd_tokens,
    )

    logger.info("Model config for seq2seq: %s", str(config))


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


    # prepare model.
    if args.model_recovery_dir:
        model_config = AutoConfig.from_pretrained(args.model_recovery_dir)
        logger.info("Read decoding model config from: %s" % model_config)

        args.model_path = os.path.join(args.model_recovery_dir, "pytorch_model.bin")
        model = LayoutlmReorderingSensegTrans.from_pretrained(args.model_path, config=model_config)
    else:
        model = LayoutlmReorderingSensegTrans.from_pretrained(
            pretrained_model_name_or_path=args.model_path, base_model_type=args.base_model_type, config=model_config,
        )

    return model, tokenizer


def prepare_for_training(args, model, checkpoint_state_dict, amp):
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [ 
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

    if amp:
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
        if checkpoint_state_dict:
            amp.load_state_dict(checkpoint_state_dict['amp'])

    if checkpoint_state_dict:
        optimizer.load_state_dict(checkpoint_state_dict['optimizer'])
        model.load_state_dict(checkpoint_state_dict['model'])

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    return model, optimizer

def loss_scale_and_bkw(args, loss, optimizer, amp, retain_graph):
    if args.n_gpu > 1:
        loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training

    if args.fp16:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward(retain_graph=retain_graph)
    else:
        loss.backward(retain_graph=retain_graph)

def train(args, training_features, model, tokenizer, tgt_tokenizer):
    """ Train the model. """

    if args.local_rank in [-1, 0] and args.log_dir:
        tb_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        tb_writer = None

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
    else:
        amp = None

    # model recover
    existing_max_step_model_optim = utils.get_max_step_model_optim(args.output_dir)
    if existing_max_step_model_optim:
        recover_step, existing_model_dir, existing_optim_path = existing_max_step_model_optim

        checkpoint_state_dict = {}
        existing_model_path = glob.glob(os.path.join(existing_model_dir, "*.bin"))[0]
        checkpoint_state_dict["model"] = torch.load(existing_model_path)
        checkpoint_state_dict["optimizer"] = torch.load(existing_optim_path)["optimizer"]
        checkpoint_state_dict["lr_scheduler"] = torch.load(existing_optim_path)["lr_scheduler"]
        if args.fp16:
            checkpoint_state_dict["amp"] = torch.load(existing_optim_path)["amp"]
    else:
        recover_step = None
        checkpoint_state_dict = None


    model.to(args.device)
    model, optimizer = prepare_for_training(args, model, checkpoint_state_dict, amp=amp)

    if args.n_gpu == 0 or args.no_cuda:
        per_node_train_batch_size = args.per_gpu_train_batch_size
    else:
        per_node_train_batch_size = args.per_gpu_train_batch_size * args.n_gpu

    train_batch_size = per_node_train_batch_size * (torch.distributed.get_world_size() if args.local_rank != -1 else 1)
    global_step = recover_step if recover_step else 0

    if args.num_training_steps == -1:
        args.num_training_steps = int(args.num_training_epochs * len(training_features) / train_batch_size)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.num_training_steps, last_epoch=-1)

    if checkpoint_state_dict:
        scheduler.load_state_dict(checkpoint_state_dict["lr_scheduler"])

    train_dataset = dataset.Seq2seqDatasetForLayoutlm(
        features=training_features, max_source_len=args.max_source_length,
        max_target_len=args.max_target_length, vocab_size=tokenizer.vocab_size,
        cls_id=tokenizer.cls_token_id, sep_id=tokenizer.sep_token_id, pad_id=tokenizer.pad_token_id, mask_id=tokenizer.mask_token_id, random_prob=args.random_prob, keep_prob=args.keep_prob,
        offset=train_batch_size * global_step, num_training_instances=train_batch_size * args.num_training_steps,
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

    logger.info("Check dataset:")
    for i in range(2):
        source_ids, target_ids, pseudo_target_ids, target_index, senseg_label_ids, target_sen_input_ids_list, target_sen_labels_list, num_effective_src_tokens, num_effective_tgt_tokens, num_effective_src_sen_tokens_list, num_effective_tgt_sen_tokens_list = train_dataset.__getitem__(
            i)
        logger.info("*" * 40)
        logger.info("Instance-%d" % i)
        try:
            src = [sid[0] for sid in source_ids]
            tgt = [tid[0] for tid in target_ids]
            pseudo = [pid[0] for pid in pseudo_target_ids]
        except TypeError:
            src = source_ids
            tgt = target_ids
            pseudo = pseudo_target_ids
        logger.info("Source tokens = %s" % " ".join(tokenizer.convert_ids_to_tokens(src)))
        logger.info("Target tokens = %s" % " ".join(tokenizer.convert_ids_to_tokens(tgt)))
        logger.info("Pseudo target tokens = %s" % " ".join(tokenizer.convert_ids_to_tokens(pseudo)))
        target_index_ = [str(i) for i in target_index]
        logger.info("Target index = %s" % " ".join(target_index_))
        senseg_label_ids_ = [str(i) for i in senseg_label_ids]
        logger.info("Senseg label ids = %s" % " ".join(senseg_label_ids_))
        num_sens = len(num_effective_src_sen_tokens_list)
        for j in range(num_sens):
            logger.info("-" * 20)
            logger.info("Target sen input ids = %s" % " ".join(tgt_tokenizer.convert_ids_to_tokens(target_sen_input_ids_list[j])))
            # print(target_sen_labels_list[j])
            logger.info("Target sen labels = %s" % " ".join(tgt_tokenizer.convert_ids_to_tokens(target_sen_labels_list[j])))

    logger.info("Model = %s" % str(model))

    # Train!
    logger.info("  ***** Running training *****  *")
    logger.info("  Num examples = %d", len(training_features))
    logger.info("  Num Epochs = %.2f", len(train_dataset) / len(training_features))
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Batch size per node = %d", per_node_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", train_batch_size)
    logger.info("  Total optimization steps = %d", args.num_training_steps)

    if args.num_training_steps <= global_step:
        logger.info("Training is done. Please use a new dir or clean this dir!")
    else:
        # The training features have been shuffled.
        train_sampler = SequentialSampler(train_dataset) \
            if args.local_rank == -1 else DistributedSampler(train_dataset, shuffle=False)
        data_collator = utils.BatchListToBatchTensorsCollator(
            max_source_len=args.max_source_length,
            max_target_len=args.max_target_length,
        )      
        train_dataloader = DataLoader(
            train_dataset, sampler=train_sampler,
            batch_size=per_node_train_batch_size,
            collate_fn=data_collator,
            num_workers=0)

        train_iterator = tqdm.tqdm(
            train_dataloader, initial=global_step,
            desc="Iter (loss=X.XXX, reordering_task_loss=X.XXX, senseg_task_loss=X.XXX, trans_task_loss=X.XXX, lr=X.XXXXXXX)", disable=args.local_rank not in [-1, 0])

        model.train()
        model.zero_grad()   

        # debug.
        # for param_name, param in model.named_parameters():
        #     print(f"{param_name}: {param.requires_grad}")

        for step, batch in enumerate(train_iterator):
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'source_idxys': batch[0],
                      'target_idxys': batch[1],
                      'pseudo_target_idxys': batch[2],
                      'target_index': batch[3],
                      'senseg_target_labels': batch[4],
                      'trans_decoder_input_ids': batch[5],
                      'trans_decoder_labels': batch[6],
                      'num_effective_source_tokens': batch[7],
                      'num_effective_target_tokens': batch[8],
                      'num_effective_src_sen_tokens': batch[9],
                      'num_effective_tgt_sen_tokens': batch[10],
                      'tokenizer': tokenizer,
                      }
            
            # Reordering task.
            reordering_fwd_output_dict = model.module.forward_reordering_task(
                source_idxys=inputs["source_idxys"],
                target_idxys=inputs["target_idxys"],
                target_index=inputs["target_index"],
                pseudo_target_idxys=inputs["pseudo_target_idxys"],
                num_effective_source_tokens=inputs["num_effective_source_tokens"],
                num_effective_target_tokens=inputs["num_effective_target_tokens"],
            )
            
            loss = reordering_fwd_output_dict["reordering_task_loss"]
            reordering_task_loss_item = loss.item()
            loss_scale_and_bkw(args, loss, optimizer, amp, retain_graph=True)

            # Senseg task.
            senseg_fwd_output_dict = model.module.forward_senseg_task(
                senseg_target_labels=inputs["senseg_target_labels"],
                num_effective_target_tokens=inputs["num_effective_target_tokens"],
                decoder_hidden_states=reordering_fwd_output_dict["decoder_hidden_states"],
                max_tgt_len=reordering_fwd_output_dict["max_tgt_len"],
            )
            loss = senseg_fwd_output_dict["senseg_task_loss"] * args.senseg_task_loss_relative_weight
            senseg_task_loss_item = loss.item()
            loss_scale_and_bkw(args, loss, optimizer, amp, retain_graph=True)

            # Trans task.
            trans_task_required_items = model.module.prepare_chunks_for_trans_task(
                senseg_target_labels=inputs["senseg_target_labels"],
                num_effective_src_sen_tokens=inputs["num_effective_src_sen_tokens"],
                num_effective_tgt_sen_tokens=inputs["num_effective_tgt_sen_tokens"],
                num_effective_target_tokens=inputs["num_effective_target_tokens"],
                tokenizer=inputs["tokenizer"],
                decoder_hidden_states=reordering_fwd_output_dict["decoder_hidden_states"],
                trans_decoder_input_ids=inputs["trans_decoder_input_ids"],
                trans_decoder_labels=inputs["trans_decoder_labels"],
            )
            num_chunks = trans_task_required_items["num_chunks"]
            # print(f"num_chunks: {num_chunks}")
            trans_task_loss_item = 0
            for i in range(num_chunks):
                trans_fwd_output_dict = model.module.forward_trans_task_one_chunk(
                    batch_size_sen_level=trans_task_required_items["batch_size_sen_level"],
                    src_sen_context_chunk=trans_task_required_items["src_sen_context_chunks"][i],
                    trans_decoder_input_ids_chunk=trans_task_required_items["trans_decoder_input_ids_chunks"][i],
                    trans_target_self_attn_mask_chunk=trans_task_required_items["trans_target_self_attn_mask_chunks"][i],
                    trans_cross_attn_mask_chunk=trans_task_required_items["trans_cross_attn_mask_chunks"][i],
                    trans_basic_tgt_mask_chunk=trans_task_required_items["trans_basic_tgt_mask_chunks"][i],
                    trans_decoder_labels_chunk=trans_task_required_items["trans_decoder_labels_chunks"][i],
                )
                loss = trans_fwd_output_dict["trans_task_loss_one_chunk"] * args.trans_task_relative_weight
                trans_task_loss_item += loss.item()
                loss_scale_and_bkw(args, loss, optimizer, amp, retain_graph=(i!=num_chunks-1))

            total_loss_item = reordering_task_loss_item + senseg_task_loss_item + trans_task_loss_item


            train_iterator.set_description('Iter (loss=%5.3f) (reordering_task_loss=%5.3f) (senseg_task_loss=%5.3f) (trans_task_loss=%5.3f) (lr=%9.7f)' % (total_loss_item, reordering_task_loss_item, senseg_task_loss_item, trans_task_loss_item, scheduler.get_lr()[0]))


            if args.fp16:
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            optimizer.step()
            scheduler.step()
            model.zero_grad()
            global_step += 1

            if args.local_rank in [-1, 0] and global_step % args.logging_steps == 0 and tb_writer is not None:
                tb_writer.add_scalar('train/lr', scheduler.get_lr()[0], global_step=global_step)
                tb_writer.add_scalar('train/loss', total_loss_item, global_step=global_step)
                tb_writer.add_scalar('train/reordering_task_loss', reordering_task_loss_item, global_step=global_step)
                tb_writer.add_scalar('train/senseg_task_loss', senseg_task_loss_item, global_step=global_step)
                tb_writer.add_scalar('train/trans_task_loss', trans_task_loss_item, global_step=global_step)

            if args.local_rank in [-1, 0] and args.save_steps > 0 and \
                    (global_step % args.save_steps == 0 or global_step == args.num_training_steps):
                save_path = os.path.join(args.output_dir, "ckpt-%d" % global_step)
                os.makedirs(save_path, exist_ok=True)
                model_to_save = model.module if hasattr(model, "module") else model

                model_to_save.save_pretrained(save_path)

                optim_to_save = {
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": scheduler.state_dict(),
                }
                if args.fp16:
                    optim_to_save["amp"] = amp.state_dict()
                torch.save(
                    optim_to_save, os.path.join(args.output_dir, 'optim.{}.bin'.format(global_step)))

                logger.info("Saving model checkpoint %d into %s", global_step, save_path)

    if args.local_rank in [-1, 0] and tb_writer:
        tb_writer.close()

def main():
    args = get_args()
    prepare(args)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()
        # Make sure only the first process in distributed training will download model & vocab

    # Load pretrained model and tokenizer
    model, tokenizer = get_model_and_tokenizer(args)

    if args.local_rank == 0:
        torch.distributed.barrier()
        # Make sure only the first process in distributed training will download model & vocab

    if args.cached_train_features_file is None:
        args.cached_train_features_file = os.path.join(args.output_dir, "features_train_wi_senseg_trans_task.pt")

    example_path = args.train_file if args.train_file else args.train_folder
    
    training_features = dataset.load_and_cache_layoutlm_examples(
        example_path=example_path, tokenizer=tokenizer, tgt_tokenizer=tgt_tokenizer, local_rank=args.local_rank,
        cached_features_file=args.cached_train_features_file, max_src_length=args.max_source_length,
        layout_flag=args.base_model_type == 'layoutlm', shuffle=True,
    )

    train(args, training_features, model, tokenizer, tgt_tokenizer)


if __name__ == "__main__":
    main()