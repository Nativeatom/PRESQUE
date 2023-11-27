import json
from config import percentages

def collect_human_perception(file_path):
    with open(file_path, "r") as fp:
        human_perception = json.load(fp)

    quant2perception = {}

    for quant in human_perception:
        quant_result = human_perception[quant]
        
        quant2perception[quant] = []
        
        quant_perception = [0 for _ in percentages]
        
        num_annotators = len(quant_result[0][1])
        
        for annotator_id in range(num_annotators):
            annotation_result = [x[1][annotator_id] for x in quant_result]
            percentage_idxs_selected = [i for i, x in enumerate(annotation_result) if x]
            
            if not len(percentage_idxs_selected):
                continue
                
            lower_bound, upper_bound = percentage_idxs_selected[0], percentage_idxs_selected[-1]
            
            for i in range(lower_bound, upper_bound + 1):
                quant_perception[i] += 1
            
            quant_perception_norm = [x / sum(quant_perception) for x in quant_perception]
            quant2perception[quant] = quant_perception_norm
            
    return quant2perception

if __name__ == "__main__":
    human_perception_path = "../data/hvd_quantifier_perception.json"

    quant2human_perception = collect_human_perception(human_perception_path)