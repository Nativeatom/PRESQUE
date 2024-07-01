<h1>PRESQUE: Pragmatic Reasoning for Quantifier Semantics</h1>

[[Paper]](https://arxiv.org/abs/2311.04659)  [[Huggingface]](https://huggingface.co/datasets/billli/QuRe
)<br />
<br />
Here is the repository for "Pragmatic Reasoning Unlocks Quantifier Semantics for Foundation Models" in EMNLP 2023.

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
#### Import QuRe from Huggingface
```
from datasets import load_dataset

ds = load_dataset("billli/QuRe")
```
data/hvd_quantifier_perception.json: The human perception of 5 quantifiers (no, few, some, most, all).
- Configuration: key is the quantifier, value is a list over all percentage values. Each entry in the each percentage value results represents whether the percentage value is selected by an annotator.
- Run ```python gen_human_perception.py``` for human perception result.

data/QuRe.json: The QuRe dataset
- Metadata sample
```
{
    "orig_sentence": "In order for a steel to be considered stainless it must have a Chromium content of at least 10.5%.", 
    "percentage": "10.50%", 
    "percentage_index": 0, 
    "math_expr": ">=0.105", 
    "quant_sent": "In order for a steel to be considered stainless it must have some Chromium content.", 
    "quantifier": "some", 
    "quantifier_position": 12, 
    "specificity": "unable", 
    "wiki_entity": "List of blade materials", 
    "topics": "metallurgy; steel; composition"
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
@inproceedings{li-etal-2023-pragmatic,
    title = "Pragmatic Reasoning Unlocks Quantifier Semantics for Foundation Models",
    author = "Li, Yiyuan  and
      Menon, Rakesh  and
      Ghosh, Sayan  and
      Srivastava, Shashank",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.emnlp-main.38",
    pages = "573--591",
}
```
