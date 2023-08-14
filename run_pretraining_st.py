# loss = kl_loss + euclidean_loss
# [mask] & attention mask (mask_array) for deleted token
import os
import json
import torch
import random
import logging
import numpy as np
from tqdm import tqdm
from pathlib import Path
from collections import namedtuple
from argparse import ArgumentParser
from tempfile import TemporaryDirectory

from tensorboardX import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.nn import KLDivLoss
import torch.nn.functional as F

import sys ###
sys.path.append("..") ###

from pytorch_pretrain_bert.modeling import Bert_student
from pytorch_pretrain_bert.tokenization import BertTokenizer
from pytorch_pretrain_bert.optimization import BertAdam, warmup_linear

import copy

InputFeatures = namedtuple("InputFeatures", "input_ids input_ids_pos input_mask input_mask_pos segment_ids")

log_format = '%(asctime)-10s: %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)

TEMP_DIR='./'

def convert_example_to_features(example, tokenizer, max_seq_length):
    tokens_original = example["tokens_original"]
    tokens_masked = example["tokens_masked"]
    
    segment_ids = example["segment_ids"]

    
    assert len(tokens_original) == len(segment_ids) <= max_seq_length  # The preprocessed data should be already truncated
    input_ids = tokenizer.convert_tokens_to_ids(tokens_original)
    assert len(tokens_masked) == len(segment_ids) <= max_seq_length  # The preprocessed data should be already truncated
    input_ids_pos = tokenizer.convert_tokens_to_ids(tokens_masked)

    input_array = np.zeros(max_seq_length, dtype=np.int)
    input_array[:len(input_ids)] = input_ids
    input_array_pos = np.zeros(max_seq_length, dtype=np.int)
    input_array_pos[:len(input_ids_pos)] = input_ids_pos
    
    mask_array = np.zeros(max_seq_length, dtype=np.int)
    mask_array[:len(input_ids)] = 1
    
    mask_array_pos = copy.deepcopy(input_array_pos)
    mask_array_pos[mask_array_pos!=0]=1
    mask_array_pos[input_array_pos==103]=0 ###

    segment_array = np.zeros(max_seq_length, dtype=np.int)
    segment_array[:len(segment_ids)] = segment_ids

    features = InputFeatures(input_ids=input_array,
                             input_ids_pos=input_array_pos, ###
                             input_mask=mask_array,
                             input_mask_pos=mask_array_pos, ###
                             segment_ids=segment_array,
                             )
    return features


class PregeneratedDataset(Dataset):
    def __init__(self, training_path, epoch, tokenizer, num_data_epochs, reduce_memory=False, mode='train'):
        self.epoch = epoch
        self.tokenizer = tokenizer
        self.vocab = tokenizer.vocab
        self.data_epoch = epoch % num_data_epochs
        data_file = training_path / f"epoch_{self.data_epoch}.json"
        metrics_file = training_path / f"epoch_{self.data_epoch}_metrics.json"
        assert data_file.is_file() and metrics_file.is_file()
        metrics = json.loads(metrics_file.read_text())
        num_samples = metrics['num_training_examples']
        if mode == 'train':
            # Samples for one epoch should not larger than 26000000
            if num_samples > 26000000:
                num_samples = 26000000
        else:
            num_samples = 1000 # NOT USE
        self.temp_dir = None
        self.working_dir = None
        seq_len = metrics['max_seq_len']
        if reduce_memory:
            self.temp_dir = TemporaryDirectory(dir=TEMP_DIR)
            self.working_dir = Path(self.temp_dir.name)
            input_ids = np.memmap(filename=self.working_dir/'input_ids.memmap',
                                  mode='w+', dtype=np.int32, shape=(num_samples, seq_len))
            input_ids_pos = np.memmap(filename=self.working_dir/'input_ids_pos.memmap',
                                  mode='w+', dtype=np.int32, shape=(num_samples, seq_len))
            input_masks = np.memmap(filename=self.working_dir/'input_masks.memmap',
                                    shape=(num_samples, seq_len), mode='w+', dtype=np.int32)
            input_masks_pos = np.memmap(filename=self.working_dir/'input_masks_pos.memmap',
                                    shape=(num_samples, seq_len), mode='w+', dtype=np.int32)
            segment_ids = np.memmap(filename=self.working_dir/'segment_ids.memmap',
                                    shape=(num_samples, seq_len), mode='w+', dtype=np.int32)
        else:
            input_ids = np.zeros(shape=(num_samples, seq_len), dtype=np.int32)
            input_ids_pos = np.zeros(shape=(num_samples, seq_len), dtype=np.int32)
            input_masks = np.zeros(shape=(num_samples, seq_len), dtype=np.int32)
            input_masks_pos = np.zeros(shape=(num_samples, seq_len), dtype=np.int32)
            segment_ids = np.zeros(shape=(num_samples, seq_len), dtype=np.int32)
        logging.info(f"Loading {mode} examples for epoch {epoch}")
        with data_file.open() as f:
            instance_index = 0
            for i, line in enumerate(tqdm(f, total=num_samples, desc=f"{mode} examples")):
                if i+1 > num_samples:
                    break
                line = line.strip()
                example = json.loads(line)
                features = convert_example_to_features(example, tokenizer, seq_len)
                input_ids[instance_index] = features.input_ids
                input_ids_pos[instance_index] = features.input_ids_pos
                segment_ids[instance_index] = features.segment_ids
                input_masks[instance_index] = features.input_mask
                input_masks_pos[instance_index] = features.input_mask_pos
                instance_index += 1
        logging.info('Real num samples:{}'.format(instance_index))
        logging.info("Loading complete!")
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.input_ids = input_ids
        self.input_ids_pos = input_ids_pos
        self.input_masks = input_masks
        self.input_masks_pos = input_masks_pos
        self.segment_ids = segment_ids

    def __len__(self):
        return self.num_samples

    def __getitem__(self, item):
        return (torch.tensor(self.input_ids[item].astype(np.int64)),
                torch.tensor(self.input_ids_pos[item].astype(np.int64)),
                torch.tensor(self.input_masks[item].astype(np.int64)),
                torch.tensor(self.input_masks_pos[item].astype(np.int64)),
                torch.tensor(self.segment_ids[item].astype(np.int64)),
                )


def main():
    parser = ArgumentParser()
    parser.add_argument('--pregenerated_data', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument("--bert_model", type=str, required=True, help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--bert_model_teacher", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True, help="Choose a model from PROP, Bert_CLS, Bert_KL")
    parser.add_argument("--train_object", type=str, required=True, help="teacher or student")
    parser.add_argument("--do_lower_case", action="store_true")
    parser.add_argument("--reduce_memory", action="store_true",
                        help="Store training data as on-disc memmaps to massively reduce memory usage")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs to train for")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--save_checkpoints_steps",
                        default=10000,
                        type=int,
                        help="How often to save the model checkpoint.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                        "0 (default value): dynamic loss scaling.\n"
                        "Positive power of 2: static loss scaling value.\n")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--learning_rate",
                        default=3e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    args = parser.parse_args()

    assert args.pregenerated_data.is_dir(), \
        "--pregenerated_data should point to the folder of files made by pregenerate_training_data.py!"

    samples_per_epoch = []
    for i in range(args.epochs):
        epoch_file = args.pregenerated_data / f"epoch_{i}.json"
        metrics_file = args.pregenerated_data / f"epoch_{i}_metrics.json"
        if epoch_file.is_file() and metrics_file.is_file():
            metrics = json.loads(metrics_file.read_text())
            # Samples for one epoch should not larger than 26000000
            metrics['num_training_examples'] = metrics['num_training_examples'] if metrics['num_training_examples'] < 26000000 else 26000000
            samples_per_epoch.append(metrics['num_training_examples'])
        else:
            if i == 0:
                exit("No training data was found!")
            print(f"Warning! There are fewer epochs of pregenerated data ({i}) than training epochs ({args.epochs}).")
            print("This script will loop over the available data, but training diversity may be negatively impacted.")
            num_data_epochs = i
            break
    else:
        num_data_epochs = args.epochs

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logging.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if args.output_dir.is_dir() and list(args.output_dir.iterdir()):
        logging.warning(f"Output directory ({args.output_dir}) already exists and is not empty!")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(args.output_dir)
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    total_train_examples = 0
    for i in range(args.epochs):
        # The modulo takes into account the fact that we may loop over limited epochs of data
        total_train_examples += samples_per_epoch[i % len(samples_per_epoch)]

    num_train_optimization_steps = int(
        total_train_examples / args.train_batch_size / args.gradient_accumulation_steps)
    if args.local_rank != -1:
        num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    # Prepare model
    if args.model_name == 'Bert_teacher_student':
        model = Bert_student.from_pretrained(args.bert_model)

    if args.fp16:
        model.half()
    device = 'cuda:1'#####
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
        model = DDP(model)
    elif n_gpu > 1:
        pass

    # Prepare optimizer for two models ###
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    if args.train_object == 'student':
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)

    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps)

    global_step = 0
    # model.train()
    for epoch in range(args.epochs):
        epoch_train_dataset = PregeneratedDataset(epoch=epoch, training_path=args.pregenerated_data, tokenizer=tokenizer,
                                            num_data_epochs=num_data_epochs, reduce_memory=args.reduce_memory)
        if args.local_rank == -1:
            train_sampler = RandomSampler(epoch_train_dataset)

        else:
            # Not supported
            train_sampler = DistributedSampler(epoch_train_dataset)

        train_dataloader = DataLoader(epoch_train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

        tr_loss = 0
        tr_loss_teacher = 0
        tr_loss_cosinie = 0
        tr_loss_kl = 0
        tr_loss_euclidean = 0
        nb_tr_steps = 0
        logging.info("***** Running training *****")
        logging.info(f"  Num examples = {total_train_examples}")
        logging.info("  Batch size = %d", args.train_batch_size)
        logging.info("  Num steps = %d", num_train_optimization_steps)
        with tqdm(total=len(train_dataloader), desc=f"Epoch {epoch}") as pbar:
            for step, batch in enumerate(train_dataloader):
                
                cosine_loss = None
                kl_loss = None
                euclidean_loss = None
                teacher_loss = None
                                
                if args.train_object == 'student':
                    batch = tuple(t.to(device) for t in batch)
                    input_ids, input_ids_pos, input_mask, input_mask_pos, segment_ids = batch
                    
                    model.train()                    
                    model_output = model(input_ids, segment_ids, input_mask)
                    student_cls = model_output['cls'] ### [batch_size, hidden_size]
                    
                    ### [batch_size, num_attention_heads, seq_len, seq_len]
                    student_attention = model_output['self_attention_matrix']
                    ### -> [batch_size * num_attention_heads * seq_len, seq_len]
                    seq_len = student_attention.shape[-1]
                    student_attention_flat = student_attention.view(-1, seq_len)
                    
                    with torch.no_grad():

                        model_output_pos = model(input_ids_pos, segment_ids, input_mask_pos) #####
                        teacher_cls = model_output_pos['cls']
                        
                        teacher_attention = model_output_pos['self_attention_matrix']
                        teacher_attention_flat = teacher_attention.view(-1, seq_len)
                        
                    cosine_loss = torch.cosine_similarity(teacher_cls, student_cls) ### [batch_size]
                    cosine_loss = 1 - torch.mean(cosine_loss) ###
                    
                    euclidean_loss = torch.mean(F.pairwise_distance(student_cls, teacher_cls, p=2), dim=0)
                    
                    def forward_plus_reverse_KL_divergence(p, q):
                        kl_loss_mean = KLDivLoss(reduction ='batchmean')
                        return (kl_loss_mean(torch.log(torch.clamp(p, 1e-20)), q) + kl_loss_mean(torch.log(torch.clamp(q, 1e-20)), p))

                    kl_loss = forward_plus_reverse_KL_divergence(student_attention_flat, teacher_attention_flat)

                    if torch.isnan(kl_loss):
                        return
                    
                    loss = kl_loss + euclidean_loss

                if n_gpu > 1:
                    pass

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                    if teacher_loss is not None:
                        teacher_loss = teacher_loss / args.gradient_accumulation_steps
                    if cosine_loss is not None:
                        cosine_loss = cosine_loss / args.gradient_accumulation_steps
                    if kl_loss is not None:
                        kl_loss = kl_loss / args.gradient_accumulation_steps
                    if euclidean_loss is not None:
                        euclidean_loss = euclidean_loss / args.gradient_accumulation_steps
                        
                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()
                    grad_list = [p.grad for n,p in param_optimizer] 
                    grad_norm = torch.sum(torch.stack([torch.norm(grad_i.detach(), 2) for grad_i in grad_list if grad_i is not None]))
                    writer.add_scalar('grad_norm', round(grad_norm.item(),4), global_step)

                tr_loss += loss.item()
                if teacher_loss is not None:
                    tr_loss_teacher += teacher_loss.item()
                if cosine_loss is not None:
                    tr_loss_cosinie += cosine_loss.item()
                if kl_loss is not None:
                    tr_loss_kl += kl_loss.item()
                if euclidean_loss is not None:
                    tr_loss_euclidean += euclidean_loss.item()
                nb_tr_steps += 1
                pbar.update(1)
                mean_loss = tr_loss * args.gradient_accumulation_steps / nb_tr_steps
                if teacher_loss is not None:
                    mean_loss_teacher = tr_loss_teacher * args.gradient_accumulation_steps / nb_tr_steps
                if cosine_loss is not None:
                    mean_loss_cosinie = tr_loss_cosinie * args.gradient_accumulation_steps / nb_tr_steps
                if kl_loss is not None:
                    mean_loss_kl = tr_loss_kl * args.gradient_accumulation_steps / nb_tr_steps
                if euclidean_loss is not None:
                    mean_loss_euclidean = tr_loss_euclidean * args.gradient_accumulation_steps / nb_tr_steps

                pbar.set_postfix_str(f"Loss: {mean_loss:.5f}")
                writer.add_scalar('train/loss', round(mean_loss,4), global_step)
                if teacher_loss is not None:
                    writer.add_scalar('train/teacher_loss', round(mean_loss_teacher,4), global_step)
                if cosine_loss is not None:
                    writer.add_scalar('train/cosine_loss', round(mean_loss_cosinie,4), global_step)
                if kl_loss is not None:
                    writer.add_scalar('train/kl_loss', round(mean_loss_kl,4), global_step)
                if euclidean_loss is not None:
                    writer.add_scalar('train/euclidean_loss', round(mean_loss_euclidean,4), global_step)


                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        # modify learning rate with special warm up BERT uses
                        # if args.fp16 is False, BertAdam is used that handles this automatically
                        lr_this_step = args.learning_rate * warmup_linear(global_step/num_train_optimization_steps,
                                                                          args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
                    if global_step % args.save_checkpoints_steps == 0:
                        with torch.no_grad():
                            # Save a ckpt for student, not for two models ###
                            logging.info("** ** * Saving model ** ** * ")
                            if args.train_object == 'student':
                                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                                output_model_file = args.output_dir / "pytorch_model_{}.bin".format(global_step)

                            torch.save(model_to_save.state_dict(), str(output_model_file))

    # Save the last model for student, not for two models ###
    logging.info("** ** * Saving model ** ** * ")
    if args.train_object == 'student':
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        output_model_file = args.output_dir / "pytorch_model_last.bin"

    torch.save(model_to_save.state_dict(), str(output_model_file))
    writer.close()

if __name__ == '__main__':
    main()
