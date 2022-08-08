
import logging
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, matthews_corrcoef
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange

from datasets import my_collate, my_collate_elmo, my_collate_pure_bert, my_collate_bert
from transformers import AdamW
from transformers import BertTokenizer

def get_collate_fn(args):
    embedding_type = args.embedding_type
    if embedding_type == 'glove':
        return my_collate
    elif embedding_type == 'elmo':
        return my_collate_elmo
    else:
        if args.pure_bert:
            return my_collate_pure_bert
        else:
            return my_collate_bert

def evaluate(args, eval_dataset, model):
    results = {}

    args.eval_batch_size = args.per_gpu_eval_batch_size
    eval_sampler = SequentialSampler(eval_dataset)
    collate_fn = get_collate_fn(args)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,
                                 batch_size=args.eval_batch_size,
                                 collate_fn=collate_fn)

    # Eval
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    for batch in eval_dataloader:
    # for batch in tqdm(eval_dataloader, desc='Evaluating'):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs, labels = get_input_from_batch(args, batch)

            logits = model(**inputs)
            tmp_eval_loss = F.cross_entropy(logits, labels)

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = labels.detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(
                out_label_ids, labels.detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=1)
    # print(preds)
    result = compute_metrics(preds, out_label_ids)
    results.update(result)

    output_eval_file = os.path.join(args.output_dir, 'eval_results_purebert_droput0.3_20220802.txt')
    with open(output_eval_file, 'a+') as writer:
        logger.info('***** Eval results *****')
        logger.info("  eval loss: %s", str(eval_loss))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("  %s = %s\n" % (key, str(result[key])))
            writer.write('\n')
        writer.write('\n')
    return results, eval_loss