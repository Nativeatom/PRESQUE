<h1>PRESQUE: Pragmatic Reasoning for Quantifier Semantics</h1>

Here is the repository for "Pragmatic Reasoning Unlocks Quantifier Semantics for Foundation Models" [[link]](https://arxiv.org/abs/2311.04659)

### Environment
```
conda env create -f presque.yaml
conda activate presque
```

### PRESQUE
#### Sample command
```
cd code
python presque.py --data_file=../data/QuRe.json  --approximate_window=2 --range_window=2 --display_every=100 --model_version=ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli
```

#### Parameters
- data_file: the data file.
- interval: the interval width &beta;.
- granularity: the granularity *g*.
- approximate_window: the window size *w* for the approximation operator (~).
- range_window: the window size *w* for other operators.
- out_dir: the directory to save results, default ../result.
- note: the note to save for result names.
- model_version: the model used to serve the NLI component.
- batch_size: the batch size for running experiments.
- display_every: the frequency to display sample output.
- device: whether to use GPU, default cuda.

### Data
data/hvd_quantifier_perception.json: The human perception of 5 quantifiers (no, few, some, most, all).

data/QuRe.json: The QuRe dataset
- Metadata sample
```
{
    "orig_sentence": "Coconut milk contains 5% to 20% fat, while coconut cream contains around 20% to 50% fat.", "percentage": "5%-20%", 
    "percentage_index": 0, 
    "math_expr": "0.05-0.2", 
    "quant_sent": "Coconut milk contains 5% to 20% fat, while coconut cream contains moderate fat.", 
    "quantifier": "moderate", 
    "quantifier_position": 14, 
    "specificity": "unable", 
    "wiki_entity": "Coconut", 
    "topics": "food; nutrition; fat percentage"
}
```
   * orig_sentence: the original sentence appeared in Wikipedia.
   * percentage: the percentage mentioned in the orig_sentence.
   * percentage_index: the index of the mentioned percentage in the orig_sentence.
   * math_expr: the percentage expression generated.
   * quant_sent: the annotated quantified sentence.
   * quantifier_position: the position of quantifier mentioned.
   * specificity:  the difficulty of deciphering the percentage scope of the quantifier from the sentence excluding the quantifier.
   * wiki_entity: the wikipedia entity that includes <i>orig_sentence</i>.
   * topics: sentence topics.


### Reference
```
@inproceedings{li-etal-2023-presque,
    title = "Pragmatic Reasoning Unlocks Quantifier Semantics for Foundation Models",
    author = "Li, Yiyuan  and
      R. Menon, Rakesh  and
      Ghosh, Sayan  and
      Srivastava, Shashank",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
}
```