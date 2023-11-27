import os
from os.path import join
import torch
import re
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from wordfreq import word_frequency
from config import quantifiers_by_category, percentages, text_percentages, src_attr_ratios, interchangeable_quantfiers

# upgrade matplotlib to 3.1.2 or higher to avoid heatmaps being cut.
# ref: https://stackoverflow.com/questions/56942670/matplotlib-seaborn-first-and-last-row-cut-in-half-of-heatmap-plot

def get_choices(type):
    choices = None
    if type == "quantifier":
        choices = quantifiers_by_category
    elif type == "percentage":
        choices = percentages
    elif type == "text_percentage":
        choices = text_percentages
    elif type == "context":
        choices = src_attr_ratios
        choices = [str(x * 100) + "%" for x in choices]
    return choices

def generate_entailment(premise, hypothesis, model, tokenizer, max_length, device):
    start_token = "<s>"
    sep_token = "</s>"
    if type(premise) == str and type(hypothesis) == str:
        tokenized_input_seq_pair = tokenizer.encode_plus(premise, hypothesis,
                                                        max_length=max_length,
                                                        return_token_type_ids=True, truncation=True)

        input_ids = torch.Tensor(tokenized_input_seq_pair['input_ids']).long().unsqueeze(0).to(device)
        # remember bart doesn't have 'token_type_ids', remove the line below if you are using bart.
        token_type_ids = torch.Tensor(tokenized_input_seq_pair['token_type_ids']).long().unsqueeze(0).to(device)
        attention_mask = torch.Tensor(tokenized_input_seq_pair['attention_mask']).long().unsqueeze(0).to(device)

        outputs = model(input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        labels=None)
        # Note:
        # "id2label": {
        #     "0": "entailment",
        #     "1": "neutral",
        #     "2": "contradiction"
        # },

        predicted_probability = torch.softmax(outputs[0], dim=1)[0].tolist()  # batch_size only one
    else:
        nli_text_pairs = ["{} {} {} {} {} {}".format(start_token, prem, sep_token, sep_token, hypo, sep_token) for prem, hypo in zip(premise, hypothesis)]
        tokenized_input_seq_pairs = tokenizer.batch_encode_plus(nli_text_pairs,
                                                        pad_to_max_length=True,
                                                        return_token_type_ids=True)

        input_ids = torch.Tensor(tokenized_input_seq_pairs['input_ids']).long().to(device)
        # remember bart doesn't have 'token_type_ids', remove the line below if you are using bart.
        token_type_ids = torch.Tensor(tokenized_input_seq_pairs['token_type_ids']).long().to(device)
        attention_mask = torch.Tensor(tokenized_input_seq_pairs['attention_mask']).long().to(device)
        try:
            outputs = model(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            labels=None)
        except:
            outputs = model(input_ids,
                            attention_mask=attention_mask,
                            labels=None)            
        predicted_probability = torch.softmax(outputs[0], dim=1)
    return predicted_probability

def generate_summary_fig(summary, category, premise_type, hypothesis_type, 
                            premise_choices, hypothesis_choices, path, file,
                            digit_fontsize=12, ticks_fontsize=12, 
                            measures=["entail", "neutral", "contradict"], save_fig=True):
    
    if premise_choices is None:
        premise_choices = get_choices(premise_type)
    if hypothesis_choices is None:
        hypothesis_choices = get_choices(hypothesis_type)

    for measure in measures:
        statement_scores = summary

        scores_matrix = np.zeros((len(premise_choices), len(hypothesis_choices)))

        for pre_idx in range(len(premise_choices)):
            for hyp_idx in range(len(hypothesis_choices)):
                try:
                    quantifier_scores = statement_scores["{}-{}".format(premise_choices[pre_idx], hypothesis_choices[hyp_idx])][measure]
                except Exception as err:
                    continue
                scores_matrix[pre_idx, hyp_idx] = np.mean(quantifier_scores)

        width = len(hypothesis_choices)
        height_wdith_ratio = len(premise_choices) / len(hypothesis_choices)
        fig, ax = plt.subplots(figsize=(width, int(width * height_wdith_ratio)))
        im = ax.imshow(scores_matrix)

        # Show all ticks and label them with the respective list entries
        plt.xticks(np.arange(len(hypothesis_choices)), labels=hypothesis_choices, fontsize=ticks_fontsize)
        plt.yticks(np.arange(len(premise_choices)), labels=premise_choices, fontsize=ticks_fontsize)

        plt.xlabel("Hypothesis Quantification")
        plt.ylabel("Premise Quantification")

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        for i in range(len(premise_choices)):
            for j in range(len(hypothesis_choices)):
                number_display = round(scores_matrix[i, j] * 100, 2)
                color = "w" if number_display < 50 else "b"
                text = ax.text(j, i, number_display,
                            ha="center", va="center", color=color, fontsize=digit_fontsize)

        ax.set_title("{} Scores in Statement {}".format(measure, category))
        fig.tight_layout()
        if save_fig:
            plt.savefig(join(path, "{}-{}.{}.png".format(category, measure, file)))
        plt.show()

class int2word(object):
    
    less_than_20 = ["", "One", "Two", "Three", "Four", "Five", "Six",
    "Seven", "Eight", "Nine", "Ten", "Eleven", "Twelve", "Thirteen",
    "Fourteen", "Fifteen", "Sixteen", "Seventeen", "Eighteen",
    "Nineteen"]
    tens = ["","Ten", "Twenty", "Thirty", "Forty", "Fifty", "Sixty",
    "Seventy", "Eighty", "Ninety"]
    thousands = ["", "Thousand", "Million", "Billion"]
    
    def numberToWords(self, num):
        if num == 0:
            return "Zero"
        ans = ""
        i = 0
        while num > 0:
            if num % 1000 != 0:
                ans = self.helper(num % 1000) + self.thousands[i] + " " + ans
                i += 1
                num //= 1000
        return ans.strip()
    
    def helper(self, n):
        if n == 0:
             return ""
        elif n < 20:
             return self.less_than_20[n] + " "
        elif n < 100:
             return self.tens[n//10] + " " + self.helper(n % 10)
        else:
             return self.less_than_20[n // 100] + " Hundred " + self.helper(n % 100)

def get_question_ids_from_html(html_file):
    
    try: 
        from BeautifulSoup import BeautifulSoup
    except ImportError:
        from bs4 import BeautifulSoup

    question_info = {}

    parsed_html = BeautifulSoup(open(html_file).read(), "html.parser")

    fields = parsed_html.find_all('fieldset')

    for field in fields:
        q = field.find("legend")
        if q is None:
            continue

        q_text = q.text
        options = [[x['name'], x['value']] for x in field.find_all("input", attrs={"type": "radio"})]
        if len(options):
            question_info[q_text] = {}
            question_info[q_text]['id'] = options[0][0]
            question_info[q_text]['options'] = [x[1] for x in options]

    return question_info

def find_matches(pattern, sentence):
     return [(match.start(), match.end()) for match in re.finditer(pattern, sentence)]

def find_percentage_positions(target_percentage, sentence):
    # remove the ending %
    compile_pattern = rf'{target_percentage[:-1]}\d+%'
    pattern = re.compile(compile_pattern, re.IGNORECASE)
    positions = find_matches(pattern, sentence)
    return positions

def dist_normalize(dist, norm="sum"):
    if isinstance(dist, dict):
        if norm == "sum":
            denominator = sum(dist.values())
            for q, v in dist.items():
                dist[q] = v / denominator
        elif norm == "softmax":
            denominator = sum(np.exp(dist.values()))
            for q, v in dist.items():
                dist[q] = np.exp(v) / denominator  
    else:
        if norm == "sum":
            denominator = sum(dist)
            dist = dist / denominator
        elif norm == "softmax":
            denominator = sum(np.exp(dist.values()))
            dist = np.exp(dist) / denominator      
    return dist
    
def generate_lh_seq(interval):
    # the likelihoods are splitted in the interval
    num_intervals = int(1 / interval)

    all_likelihoods = [round(i * interval, 2) for i in range(num_intervals + 1)]
    all_likelihoods_str = ["{}%".format(round(x * 100), 2) for x in all_likelihoods]
    return all_likelihoods_str

def token_assembly(tokens):
    # including necessary spacing to tokenized sequences (including punctuations)
    # sample input: ['Its', 'flavour', 'comes', 'from', 'geraniol', '(', '3', '–', '40', '%),', 'neral', '(', '3', '–', '35', '%),', 'geranial', '(', '4', '–', '85', '%)', '(', 'both', 'isomers', 'of', 'citral', '),', '[some]', '(', 'E', ')-', 'caryophyllene', ',', 'and', 'citronellal', '(', '1', '–', '44', '%).']
    # sample output: Its flavour comes from geraniol (3–40%), neral (3–35%), geranial (4–85%) (both isomers of citral), [some] (E)-caryophyllene, and citronellal (1–44%). 
    
    assembly_tokens = []

    num_tokens = len(tokens)

    previous_token = None

    quote_pair = 0
    
    for idx in range(num_tokens):
        add_space = True
        curr_token = tokens[idx]
        previous_token = curr_token
        if curr_token.startswith("–") or "-" in curr_token:
            # remove previous and proceeding space
            if assembly_tokens[-1] == " ":
                assembly_tokens.pop()
            add_space = False
            
        if curr_token.startswith("%"):
            if idx > 0 and tokens[idx-1].isdigit():
                # decimals
                if assembly_tokens[-1] == " " :
                    assembly_tokens.pop() 
                    
        if curr_token.startswith("."):
            if assembly_tokens[-1] == " ":
                assembly_tokens.pop()
                assembly_tokens.append(curr_token)
                if idx < num_tokens - 1 and tokens[idx+1].isdigit():
                    # skipping comma as number separators
                    continue
                assembly_tokens.append(" ")
                continue
            add_space = False
            
        if curr_token.startswith(","):
            if assembly_tokens[-1] == " ":
                assembly_tokens.pop()
                if idx < num_tokens - 1 and tokens[idx+1].isdigit():
                    # skipping comma as number separators
                    add_space = False
                
        if curr_token.startswith('"'):
            if quote_pair:
                # remove the previous space
                if assembly_tokens[-1] == " " :
                    assembly_tokens.pop() 
            else:
                # skip the proceeding space
                add_space = False
            quote_pair = not quote_pair
            
        if ")" in curr_token and assembly_tokens[-1] == " ":
            assembly_tokens.pop()
            
        assembly_tokens.append(curr_token)
        
        add_space = False if "(" in curr_token[0] else add_space
        
        if add_space:
            assembly_tokens.append(" ")
    return assembly_tokens

def compute_prob_boundary(math_expr, granularity, approximate_window, range_window, display_result=False):
    # granularity: the smallest unit that could change in llh
    # approximate_window: the approximate window for ~: i.e. ~X: Uniform[X-window, X+window]
    # range_window: the window for </>/<=: i.e. <X: Uniform[X-window, X-1]
    # return: a dictionary storing the quantifier and the spectrum as a list
    
    max_llh_idx = int(1 / granularity) - 1
    likelihood_spectrum = np.array([0 for _ in range(int(1 / granularity))]).astype(float)
    
    low_bound_prob = 0
    upper_bound_prob = 1
    
    math_expr = math_expr.lstrip("'")

    if math_expr.startswith("~"):
        # approximation
        math_expr_pure = float(math_expr.lstrip("~"))
        math_expr_ground_id = int(math_expr_pure // granularity)
        start_idx, end_idx = max(0, math_expr_ground_id - approximate_window), min(max_llh_idx, math_expr_ground_id + approximate_window)
        return max(low_bound_prob, start_idx * granularity), min(upper_bound_prob, end_idx * granularity)

    elif math_expr.startswith("<"):
        # no greater than/less than
        if math_expr.startswith("<="):
            math_expr_pure = float(math_expr.lstrip("<="))
            math_expr_ground_id = int(math_expr_pure // granularity)
        else:
            math_expr_pure = float(math_expr.lstrip("<"))
            math_expr_ground_id = int(math_expr_pure // granularity) - 1               
        start_idx = max(0, math_expr_ground_id - range_window) if range_window != "max" else 0
        end_idx = min(max_llh_idx, math_expr_ground_id)
        return max(low_bound_prob, start_idx * granularity), min(upper_bound_prob, end_idx * granularity)

    elif math_expr.startswith(">"):
        # no less than/greater than
        if math_expr.startswith(">="):
            math_expr_pure = float(math_expr.lstrip(">="))
            math_expr_ground_id = int(math_expr_pure // granularity)
        else:
            math_expr_pure = float(math_expr.lstrip(">"))
            math_expr_ground_id = int(math_expr_pure // granularity) + 1
        start_idx = math_expr_ground_id
        end_idx = min(max_llh_idx, math_expr_ground_id + range_window) if range_window != "max" else max_llh_idx
        return max(low_bound_prob, start_idx * granularity), min(upper_bound_prob, end_idx * granularity)

    elif "-" in math_expr:
        # range
        interval = [float(x.strip()) for x in math_expr.split("-")]
        start_idx = int(interval[0] // granularity)
        end_idx = int(interval[1] // granularity)
        return max(low_bound_prob, start_idx * granularity), min(upper_bound_prob, end_idx * granularity)

    else:
        # exact value
        math_expr_pure = float(math_expr.lstrip("'").strip("."))
        math_expr_ground_id = int(math_expr_pure // granularity)
        new_prob = math_expr_ground_id * granularity
        return new_prob, new_prob


def create_sent_template(data_entry, tokenizer, quantifiers):
    original_sentence = data_entry["orig_sentence"]
    original_sentence = original_sentence.replace('–', '-')
    quant_sent = data_entry['quant_sent']
    quantifier = data_entry['quantifier']
    quantifier_start = quantifier[0].upper() + quantifier[1:]
    quant_position = data_entry['quantifier_position']
    tokens = tokenizer.tokenize(quant_sent)
    if quant_sent.count(quantifier) == 1:
        quant_sent_template = quant_sent.replace(quantifier, "[Q]")
    elif quant_sent.count(quantifier_start) == 1:
        quant_sent_template = quant_sent.replace(quantifier_start, "[Q]")
    else:
        quant_position_in_tokens = [i for i, t in enumerate(tokens) if t in [quantifier, quantifier_start]]
        closet_quant_position = sorted(quant_position_in_tokens, key=lambda x: abs(x-quant_position))
        if len(closet_quant_position):
            tokens[closet_quant_position[0]] = "[Q]"
        else:
            tokens[quant_position] = "[Q]"
        quant_sent_template = "".join(token_assembly(tokens))

    percentage = data_entry["percentage"]
    percentage_matches = []
    percent_patterns = []
    linking_words = ["-", "to", "and"]
    decimal_pattern = "([0-9]+).([0-9]+)%"
    option = "None"

    if percentage in original_sentence:
        percentage_matches = [(match.start(), match.end()) for match in re.finditer(percentage, original_sentence)]

    elif percentage.replace("0", "") in original_sentence:
        percentage_matches = [(match.start(), match.end()) for match in re.finditer(percentage.replace("0", ""), original_sentence)]

    elif '-' in percentage:
        for linking_word in linking_words:
            percent_patterns.append(percentage.replace('-', " {} ".format(linking_word)))
            percent_patterns.append(percentage.replace('-', "0{}".format(linking_word)))
            percent_patterns.append(percentage.replace('-', "0 {}".format(linking_word)))
            percent_patterns.append(percentage.replace('-', "0% {} ".format(linking_word)))
            percent_patterns.append(percentage.replace('%-', "{}".format(linking_word)))
            percent_patterns.append(percentage.replace('%-', " {} ".format(linking_word)))
            percent_patterns.append(percentage.replace('%-', "0{}".format(linking_word)))
            percent_patterns.append(percentage.replace('%-', "0 {} ".format(linking_word)))
            percent_patterns.append(percentage.replace('%-', "0%{}".format(linking_word)))
            percent_patterns.append(percentage.replace('%-', "0% {} ".format(linking_word)))
            
        for percent_pattern in percent_patterns:
            if percent_pattern in original_sentence:
                percentage_matches = find_matches(percent_pattern, original_sentence)
                break

    # ending 0%. e.g. 10.50%
    elif not len(percentage_matches) and "." in percentage:
        percentage_depart = re.match(decimal_pattern, percentage)
        if percentage_depart is not None:
            nondeci, deci = percentage_depart.groups()
            # prefix match for shorter decimal digits, e.g. 0.11% to 0.112%
            percentage_matches = find_percentage_positions(percentage, original_sentence)
            if not len(percentage_matches):
                if deci.endswith("00"):
                    norm_deci = deci.replace("00", "0")
                    norm_percentage = "{}.{}%".format(nondeci, norm_deci)
                    percentage_matches = find_matches(norm_percentage, original_sentence)
                elif deci.endswith("0"):
                    norm_deci = deci.rstrip("0")
                    norm_percentage = "{}.{}%".format(nondeci, norm_deci)
                    percentage_matches = find_matches(norm_percentage, original_sentence)
                    
    if len(percentage_matches):
        percentage_start_appr = sum([len(x) + 1 for x in tokens[:quant_position]]) - 1
        nearest_matches = [abs(x[0] - percentage_start_appr) for x in percentage_matches]
        percentage_nearest_span = percentage_matches[np.argmin(nearest_matches)]
        start, end = percentage_nearest_span
        percent_sent_template = original_sentence[:start] + "[P]" + original_sentence[end:]
    return quant_sent_template, percent_sent_template

def compute_consecutiveness(sort_list):
    # sort_list: list
    n = len(sort_list) - 1
    return (sum(np.diff(sorted(sort_list)) == 1) >= n)

def compute_interval_distance(pred_buckets, lower_bound, upper_bound):
    if isinstance(pred_buckets, list):
        return np.mean([compute_interval_distance(pred_bucket, lower_bound, upper_bound) for pred_bucket in pred_buckets])
    else:
        return 0 if lower_bound <= pred_buckets and upper_bound >= pred_buckets else min(abs(lower_bound - pred_buckets), abs(upper_bound - pred_buckets))

def computing_maximum_lh_spans(scores, topk):
    # scores is an unsorted list
    topk_ids = np.argsort(scores)[::-1][:topk]

    all_spans = find_spans(topk_ids)

    span_scores = []

    for span in all_spans:
        start, end = span
        span_scores.append((span, sum(scores[start:end+1])))

    span_scores = sorted(span_scores, key=lambda x:x[1], reverse=True)
    return span_scores[0]

def find_spans(cand_ids):
    # example input: [2, 4, 5, 6, 7, 8, 10, 11, 12, 14, 16]
    # example output: [(2, 2), (4, 8), (10, 12), (14, 14)]
    cand_ids = sorted(cand_ids)
    spans = []

    start_idx = 0
    for idx in range(len(cand_ids)):
        if idx and cand_ids[idx] > cand_ids[idx - 1] + 1:
            spans.append((cand_ids[start_idx], cand_ids[idx - 1]))
            start_idx = idx

    new_span = (cand_ids[start_idx], cand_ids[idx]) if start_idx != idx else (cand_ids[idx], cand_ids[idx])
    spans.append(new_span)
    return spans


def create_quant_set():
    interchangeable_set = {}
    quantifier_freq = {}

    for qs in interchangeable_quantfiers:
        for q in qs:
            if q not in interchangeable_set:
                interchangeable_set[q] = []
                quantifier_freq[q] = word_frequency(q, 'en')
            for oqs in interchangeable_quantfiers:
                if q in oqs:
                    interchangeable_set[q] += [x for x in oqs if x != q]

    return interchangeable_set, quantifier_freq