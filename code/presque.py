import os
from os.path import join
import re
import json
import numpy as np
import pandas as pd
import csv
import random
from tqdm import tqdm
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from nltk.tokenize import WordPunctTokenizer
from wordfreq import word_frequency
import torch
from utils import *
import argparse

def evaluation(input_seqs, model, tokenizer, bz, device):
    all_outputs = []

    batch_id = 0
    num_instance = len(input_seqs)

    input_seq_tokens = [len(x.split()) for x in input_seqs]

    while batch_id * bz < num_instance:
        tokenized_input_seq_pairs = tokenizer.batch_encode_plus(input_seqs[batch_id * bz : (batch_id + 1) * bz],
                                                                pad_to_max_length=True,
                                                                return_token_type_ids=True)
        try:
            input_ids = torch.Tensor(tokenized_input_seq_pairs['input_ids']).long().to(device)
            # remember bart doesn't have 'token_type_ids', remove the line below if you are using bart.
            token_type_ids = torch.Tensor(tokenized_input_seq_pairs['token_type_ids']).long().to(device)
            attention_mask = torch.Tensor(tokenized_input_seq_pairs['attention_mask']).long().to(device)
            outputs = model(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            labels=None)  
            all_outputs.append(outputs[0])     
            del input_ids, token_type_ids, attention_mask    
        except torch.cuda.OutOfMemoryError:
            for i in range(len(tokenized_input_seq_pairs)):
                input_ids = torch.Tensor(tokenized_input_seq_pairs['input_ids'][i]).long().unsqueeze(0).to(device)
                # remember bart doesn't have 'token_type_ids', remove the line below if you are using bart.
                token_type_ids = torch.Tensor(tokenized_input_seq_pairs['token_type_ids'][i]).long().unsqueeze(0).to(device)
                attention_mask = torch.Tensor(tokenized_input_seq_pairs['attention_mask'][i]).long().unsqueeze(0).to(device)
                outputs = model(input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids,
                                labels=None)  
                all_outputs.append(outputs[0])    
                del input_ids, token_type_ids, attention_mask
        except:
            outputs = model(input_ids,
                            attention_mask=attention_mask,
                            labels=None)  
            all_outputs.append(outputs[0])
        batch_id += 1
        torch.cuda.empty_cache()
        
    
    # dim: num_inputs, 3
    all_outputs = torch.cat(all_outputs, dim=0)
    return all_outputs  

def quantifier_entailment(data, model, tokenizer, template_tokenizer, 
                            prob_ground_setup, configuration):

    interval = prob_ground_setup['interval']
    granularity = prob_ground_setup['granularity']
    approximate_window = prob_ground_setup['approximate_window']
    range_window = prob_ground_setup['range_window']

    interchangeable_set = configuration['quant_equivalent_set']
    quantifiers = configuration['quantifiers']
    batch_size = configuration["batch_size"]
    quant_prior = configuration["quantifier_prior"]
    consecutive_ks = configuration['consecutiveness_k']
    interval_dist_ks = configuration['interval_dist_k']
    display_every = configuration['display_every']

    prior_config = configuration["prior"]

    quant_interchangeable = configuration["quant_interchangeable"]

    device = configuration["device"]

    prob_seqs = generate_lh_seq(interval)

    metric_by_infer_lh = {}

    metrics = ["Recall@1", "MRR", "CrossEntropy"]

    metrics += ["Consecutiveness@{}".format(k) for k in consecutive_ks]
    metrics += ["IntervalDistance@{}".format(k) for k in interval_dist_ks]

    results = []

    quantifier_list = [""]

    MemorySkipped_idxes = []

    evaluations = []

    for idx in tqdm(range(len(data))):
        data_entry = data[idx]
        
        quant_template, percent_tempalte = create_sent_template(data_entry, template_tokenizer, quantifiers)

        quant_sent = data_entry['quant_sent']
        quantifier = data_entry['quantifier']
        math_expr = data_entry['math_expr']
        infer_lh = data_entry['specificity']

        if infer_lh not in metric_by_infer_lh:
            metric_by_infer_lh[infer_lh] = {"L0": {}, "L1": {}}
            for metric in metrics:
                metric_by_infer_lh[infer_lh]["L0"][metric] = []
                metric_by_infer_lh[infer_lh]["L1"][metric] = []

        prob_lower, prob_upper = compute_prob_boundary(math_expr, granularity, approximate_window, range_window)

        bucket_lower = int(np.floor(prob_lower / interval))
        bucket_upper = int(np.ceil(prob_upper / interval))

        # literal listener
        premise = []
        hypothesis = []
        
        for prob_seq in prob_seqs:
            percent_sent = percent_tempalte.replace("[P]", prob_seq)
            premise.append(quant_sent)
            hypothesis.append(percent_sent)

        start_token = "<s>"
        sep_token = "</s>"

        literal_lisenter_nli_text_pairs = ["{} {} {} {} {}".format(start_token, prem, sep_token, hypo, sep_token) for prem, hypo in zip(premise, hypothesis)]
        literal_speaker_nli_text_pairs = ["{} {} {} {} {}".format(start_token, prem, sep_token, hypo, sep_token) for prem, hypo in zip(hypothesis, premise)]

        splits = [len(literal_lisenter_nli_text_pairs), len(literal_speaker_nli_text_pairs)]

        try:
            all_outputs = evaluation(literal_lisenter_nli_text_pairs + literal_speaker_nli_text_pairs, 
                                        model, tokenizer, batch_size, device)
            # softmax over the enatilment normalization (o/w negative values)
            predicted_probability = torch.softmax(all_outputs[:splits[0]], dim=1)
        except:
            MemorySkipped_idxes.append(idx)
            print("idx={} entaillment skipped".format(idx))
            continue

        entailments = predicted_probability[:, 0]
        literal_pred_buckets = torch.argsort(entailments).cpu().numpy().tolist()[::-1] if device != "cpu" else torch.argsort(entailments).numpy().tolist()[::-1]

        # L_0(p|q_0)
        literal_listener_probs = entailments / entailments.sum()
        literal_listener_cross_entropy = -torch.log(literal_listener_probs)[bucket_lower:bucket_upper+1].sum().item()
        del predicted_probability

        # S_0(q|p) with q being quantifier
        pragmatic_entailments = all_outputs[splits[0]:splits[0]+splits[1]]
        # sum_{p} S_0(q|p) = 1
        # pragmatic_probs = pragmatic_entailments / pragmatic_entailments.sum()
        pragmatic_probs = torch.softmax(pragmatic_entailments, dim=1)[:, 0]
        del all_outputs

        # literal speaker
        if quantifier in quant_sent:
            quantifier_in_sent = quantifier
        elif quantifier + "ly" in quant_sent:
            quantifier_in_sent = quantifier + "ly"

        quantifier_candidates = [quantifier_in_sent] + interchangeable_set[quantifier_in_sent] if quant_interchangeable else quantifiers

        prior_premise = []
        prior_hypothesis = []

        prior_splits = []

        quant = quantifier_in_sent
        for prob_seq in prob_seqs:
            prior_quant_sent = quant_template.replace("[Q]", quant)
            prior_percent_sent = percent_tempalte.replace("[P]", prob_seq)
            prior_premise.append(prior_quant_sent)
            prior_hypothesis.append(prior_percent_sent)    
        prior_splits.append(len(prob_seqs))

        literal_lisenter_prior_nli_text_pairs = ["{} {} {} {} {}".format(start_token, prem, sep_token, hypo, sep_token) for prem, hypo in zip(prior_premise, prior_hypothesis)]   

        all_outputs_prior = evaluation(literal_lisenter_prior_nli_text_pairs, model, tokenizer, batch_size, device)

        # compute conditional probability by normalization
        num_start = 0
        p_cond_qs = []
        p_joint_qs = []
        for split_id, split in enumerate(prior_splits):
            # q_0
            quant_select = quantifier_candidates[split_id] if quant_interchangeable else quantifier_in_sent
            # normalize the entailment/neutral/contradict scores
            quant_probs = torch.softmax(all_outputs_prior[num_start:num_start+split], dim=1)

            # P(p|q_0)
            p_cond_qs.append(dist_normalize(quant_probs[:, 0].detach().cpu().numpy(), norm="sum"))
            p_joint_qs.append(p_cond_qs[-1] * quant_prior[quant_select])
            num_start += split
        del all_outputs_prior
        
        p_joint_qs = np.stack(p_joint_qs)

        # P(p)
        p_prior = p_joint_qs.sum(0)
        # normalize P(p)
        p_prior = p_prior / p_prior.sum()

        # L_1(p|q0)
        if prior_config == "contextual":
            pragmatic_listener = pragmatic_probs.cpu().detach().numpy() * p_prior
        elif prior_config == "uniform":
            pragmatic_listener = pragmatic_probs.cpu().detach().numpy()
        # normalization
        pragmatic_listener = pragmatic_listener / pragmatic_listener.sum()

        pragmatic_pred_buckets = np.argsort(pragmatic_listener)[::-1].tolist()
        pragmatic_listener_cross_entropy = -sum([np.log(x) for x in pragmatic_listener[bucket_lower:bucket_upper+1]])

        metric_by_infer_lh[infer_lh]["L0"]["Recall@1"].append(literal_pred_buckets[0] == bucket_lower or literal_pred_buckets[0] == bucket_upper)
        metric_by_infer_lh[infer_lh]["L1"]["Recall@1"].append(pragmatic_pred_buckets[0] == bucket_lower or pragmatic_pred_buckets[0] == bucket_upper)

        literal_avg_rank = np.mean([literal_pred_buckets.index(x) for x in range(bucket_lower, bucket_upper + 1)]) + 1 if bucket_lower < bucket_upper else literal_pred_buckets.index(bucket_lower) + 1
        metric_by_infer_lh[infer_lh]["L0"]["MRR"].append(1 / literal_avg_rank)
        pragmatic_avg_rank = np.mean([pragmatic_pred_buckets.index(x) for x in range(bucket_lower, bucket_upper + 1)]) + 1 if bucket_lower < bucket_upper else pragmatic_pred_buckets.index(bucket_lower) + 1
        metric_by_infer_lh[infer_lh]["L1"]["MRR"].append(1 / pragmatic_avg_rank)

        metric_by_infer_lh[infer_lh]["L0"]["CrossEntropy"].append(literal_listener_cross_entropy)
        metric_by_infer_lh[infer_lh]["L1"]["CrossEntropy"].append(pragmatic_listener_cross_entropy)

        for consecutive_k in consecutive_ks:
            metric_by_infer_lh[infer_lh]["L0"]["Consecutiveness@{}".format(consecutive_k)].append(compute_consecutiveness(literal_pred_buckets[:consecutive_k]))
            metric_by_infer_lh[infer_lh]["L1"]["Consecutiveness@{}".format(consecutive_k)].append(compute_consecutiveness(pragmatic_pred_buckets[:consecutive_k]))
        
        for interval_dist_k in interval_dist_ks:
            metric_by_infer_lh[infer_lh]["L0"]["IntervalDistance@{}".format(interval_dist_k)].append(compute_interval_distance(literal_pred_buckets[:interval_dist_k], 
                                                                                                                               bucket_lower, bucket_upper))
            metric_by_infer_lh[infer_lh]["L1"]["IntervalDistance@{}".format(interval_dist_k)].append(compute_interval_distance(pragmatic_pred_buckets[:interval_dist_k], 
                                                                                                                               bucket_lower, bucket_upper))
        
        literal_result = {
            "probility_ground_setup": prob_ground_setup,
            "task": "L0",
            "model": configuration['model_version'],
            "orig_sentence": data_entry['orig_sentence'],
            "math_expr": math_expr,
            "lower_prob": prob_lower,
            "upper_prob": prob_upper,
            "lower_bucket": bucket_lower,
            "upper_bucket": bucket_upper,
            "data": data_entry,
            "pred_scores": entailments.detach().cpu().numpy().tolist(), # raw scores
            "L_0(p|q_0)": literal_listener_probs.detach().cpu().numpy().tolist(),
            "ranking": literal_pred_buckets,
            "cross_entropy": literal_listener_cross_entropy,
        }

        results.append(literal_result)

        pragmatic_result = {
            "probility_ground_setup": prob_ground_setup,
            "task": "L1",
            "model": configuration['model_version'],
            "orig_sentence": data_entry['orig_sentence'],
            "math_expr": math_expr,
            "lower_prob": prob_lower,
            "upper_prob": prob_upper,
            "lower_bucket": bucket_lower,
            "upper_bucket": bucket_upper,
            "data": data_entry,
            "prob_seqs": prob_seqs, # all choices of [P]s
            "quant_seqs": quantifier_candidates, # all choices of [Q]s
            "p_cond_qs": [x.tolist() for x in p_cond_qs],
            "p_prior": p_prior.tolist(),
            "pred_scores": pragmatic_entailments.detach().cpu().numpy().tolist(), # raw scores
            "S_0(q_0|p)": pragmatic_probs.detach().cpu().numpy().tolist(),
            "ranking": pragmatic_pred_buckets,
            "cross_entropy": pragmatic_listener_cross_entropy,
        }

        results.append(pragmatic_result)
        
        if idx % display_every == 0:
            print("#{} quant: {} infer: {} math_expr: {} prob: {}-{} [bucket: {}-{}] LiteralPred: {} PragmaticPred: {}".format(idx, quantifier, infer_lh,
                                                                                        math_expr, prob_lower, prob_upper, bucket_lower, bucket_upper,
                                                                                        literal_pred_buckets, pragmatic_pred_buckets))
        torch.cuda.empty_cache()
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_version", type=str, default="ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli")
    parser.add_argument("--data_file", type=str, default='../data/QuRe.json')
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--interval", type=float, default=0.1)
    parser.add_argument("--granularity", type=float, default=0.01)
    parser.add_argument("--approximate_window", type=int, default=1)
    parser.add_argument("--range_window", type=int, default=2)
    parser.add_argument("--interchangeable", type=int, default=0, help="whether to use interchangeable quantifiers")
    parser.add_argument("--max_consecutive_k", type=int, default=5)
    parser.add_argument("--max_interval_k", type=int, default=3)
    parser.add_argument("--display_every", type=int, default=20)
    parser.add_argument("--out_dir", type=str, default="../result")
    parser.add_argument("--note", type=str, default="contextual_prior")
    parser.add_argument("--save_fig", type=bool, default=False)
    args = parser.parse_args()

    interchangeable_set, quantifier_freq = create_quant_set()
                    
    # normalize over all quantifier or just the interchangeable set only has a factor difference
    quantifier_freq_norm = dist_normalize(quantifier_freq, norm='sum')

    # paths
    data_file = args.data_file
    model_version = args.model_version

    quantifiers = ['all', 'generally', 'most', 'usually', 
                    'some', 'likely', 'few', 'little', 'occasionally', 
                    'none', 'seldom', 'tiny', 'small', 'moderate', 'large']

    interval = args.interval

    granularity = args.granularity

    approximate_window = args.approximate_window

    range_window = args.range_window

    bz = args.batch_size

    consecutive_ks = [x for x in range(2, max(3, args.max_consecutive_k))]

    interval_distance_ks = [x + 1 for x in range(args.max_interval_k)]

    device = args.device

    note = args.note

    interchangeable = args.interchangeable

    annotate_version = 'QuRe'

    prob_ground_setup = {
        "interval": interval,
        "granularity": granularity,
        "approximate_window": approximate_window,
        "range_window": range_window
    }

    config = {
        "quantifiers": quantifiers,
        "model_version": model_version,
        "quantifier_prior": quantifier_freq_norm,
        "quant_equivalent_set": interchangeable_set,
        "batch_size": bz,
        "consecutiveness_k": consecutive_ks,
        "interval_dist_k": interval_distance_ks,
        "prior": "contextual",
        "quant_interchangeable": interchangeable,
        "display_every": args.display_every,
        "device": device,
    }
    
    with open(data_file, "r") as fp:
        dataset = [json.loads(x) for x in fp.readlines()]

    model_name = model_version.split("/")[-1]

    model_specific_dir = join(args.out_dir, model_name)

    if not os.path.exists(model_specific_dir):
        os.mkdir(model_specific_dir)

    out_file = "dataset={}.interval=[{}].g=[{}].appr_window={}.range_window={}.ineterchange={}.note={}.json".format(annotate_version,
                                                                          interval, granularity, approximate_window, range_window,
                                                                          bool(interchangeable), note)

    nli_tokenizer = AutoTokenizer.from_pretrained(model_version)
    nli_model = AutoModelForSequenceClassification.from_pretrained(model_version).to(device)
    punc_tokenizer = WordPunctTokenizer()
    nli_model = nli_model.eval()

    with torch.no_grad():
        results = quantifier_entailment(dataset, nli_model, nli_tokenizer, punc_tokenizer, 
                                        prob_ground_setup, config)
    
    with open(join(model_specific_dir, out_file), "w") as fp:
        for r in results:
            fp.write(json.dumps(r) + "\n")
        fp.close()




