import csv
import os

class DatasetFactory():

    @staticmethod
    def load(data="TRUE"):

        if data == "TRUE":
            return load_TRUE()
        
        #### implement your own data set here ####

        #if data == "MY_DATA":
        #    return dataset_metric, dataset_domain, domain_count, metric short
        #
        # where
        # dataset_metric: a mapping from dataset names to a dictionaries with metric 
        #                 names and their scores {metric_name: score}
        #                 where one metric_name must be "label" to specify the binary 0/1 gold scores
        #                 e.g., {"mydatasetA":{"metric1": [0.91, 0.01, 0.99], "label":[1, 0, 0]}}
        # --
        # dataset_domain: OPTIONAL (can be None), mapping from dataset names to domains;
        #                 REQUIRED when calibration mode is "indomain"
        # --
        # domain_count: OPTIONAL (can be None), mapping from domains to the count of datasets from this domain
        #               REQUIRED when calibration mode is "indomain"
        # --
        # metric_short: mapping from metric names to short names for nice presentation in output table,
        #               e.g. {"zeberta-xxl-01-v07":"zeberta"}

        raise ValueError("dataset with name {} is unknown".format(data))

def load_TRUE():
    
    path = "../data/true_data/"


    metrics = ["label", "BLEU_1","BLEU_4","ROUGE_L", "F1_overlap", "BLEURT_D6", "NUBIA", "QuestEval", "FactCC", "SummaCConv", "SummacZS", "BARTScore", "BERTScore_P_roberta","Q2","ANLI","BERTScore_P_deberta","BLEURT"]
    exclude = ["fever_dev_true_scores.csv", "vitaminc_dev_true_scores.csv"]
    metrics = ["label", "BLEU_4",  "QuestEval", "FactCC", "SummaCConv", "SummacZS", "BARTScore", "BERTScore_P_roberta","Q2", "ANLI", "BERTScore_P_deberta", "BLEURT"]
    metric_short = {"BLEU_4": "BLEU",
            "QuestEval" : "QuestE",
            "FactCC" : "FactCC",
            "SummaCConv": "SummaCC",
            "SummacZS" : "SummacZS",
            "BARTScore": "BARTSc",
            "BERTScore_P_roberta":"RoBERTSc",
            "Q2":"Q2",
            "ANLI":"ANLI",
            "BERTScore_P_deberta": "DeBERTSc",
            "BLEURT": "BLEURT"}
    
    
    domain_dataset = {"summary": ["mnbm_true_scores.csv", "frank_dev_true_scores.csv", "summeval_true_scores.csv", "qags_xsum_true_scores.csv", "qags_cnndm_true_scores.csv"],
                   "dialog": ["dialfact_dev_true_scores.csv", "begin_dev_true_scores.csv", "q2_true_scores.csv"],
                   "wiki": ["paws_wiki_dev_true_scores.csv"]}
    dataset_domain = {}
    domain_count = {do:len(domain_dataset[do]) for do in domain_dataset}
    for do in domain_dataset:
        for ds in domain_dataset[do]:
            dataset_domain[ds] = do
    
    def readf(p):
        with open(p) as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',', quotechar='"')
            for i, row in enumerate(csvreader):
                if i == 0:
                    labels = row
                    metric_score = {k:[] for k in labels}
                else:
                    for j, label in enumerate(labels):
                        if label in metrics:
                            metric_score[label].append(float(row[j]))
            return metric_score

    dataset_metric = {}
    for fi in os.listdir(path):
        if fi.endswith("csv") and fi not in exclude:
            dat = readf(path + fi)
            dataset_metric[fi] = dat
    
    return dataset_metric, dataset_domain, domain_count,  metric_short
