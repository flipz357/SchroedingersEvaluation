from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np
from random import shuffle
from sklearn.linear_model import LogisticRegression
import json
import data_helpers as dh
import argparse

def evaluate_calibrate(dataset_metric, metrics, dataset_domain=None, domain_count=None, mode=None):

    EVAL_AUC = []
    EVAL_ACC = []

    for ds in dataset_metric:
        performance_auc = list()
        performance_acc = list()
        gold = dataset_metric[ds]["label"]
        
        for i, metric in enumerate(metrics):
              
            trainpreds = []
            traingold = []
            for dsx in dataset_metric:
                if dsx == ds:
                    continue
                if mode == "indomain":
                    domain = dataset_domain[ds]
                    if domain_count[domain] > 1 and domain != dataset_domain[dsx]:
                        continue
                elif mode == "outdomain":
                    domain = dataset_domain[ds]
                    if domain == dataset_domain[dsx]:
                        continue
                trainpreds += dataset_metric[dsx][metric]
                traingold += dataset_metric[dsx]["label"]
            """ 
            traingold = []
            for dsx in dataset_metric:
                if dsx == ds:
                    continue
                if mode == "indomain":
                    domain = dataset_domain[ds]
                    if domain_count[domain] > 1 and domain != dataset_domain[dsx]:
                        continue
                elif mode == "outdomain":
                    domain = dataset_domain[ds]
                    if domain == dataset_domain[dsx]:
                        continue
                traingold += dataset_metric[dsx]["label"]
            """
            preds = dataset_metric[ds][metric] 
            predsx = [[p] for p in preds]
            trainpredsx = [[trainpreds[i]] for i in range(len(trainpreds))]
            clf = LogisticRegression()
            clf.fit(trainpredsx, traingold)
            testybar = clf.predict(predsx)

            acc = accuracy_score(gold, testybar)
            auc = roc_auc_score(gold, preds)
            
            performance_auc.append(auc)
            performance_acc.append(acc)
            

        EVAL_AUC.append(performance_auc)
        EVAL_ACC.append(performance_acc)

    return EVAL_AUC, EVAL_ACC

def evaluate_calibrate_indata(dataset_metric, metrics):

    EVAL_AUC = []
    EVAL_ACC = []

    for ds in dataset_metric:
        performance_auc = list()
        performance_acc = list()
        
        gold = dataset_metric[ds]["label"]
        idxs = list(range(len(gold)))
            

        for _ in range(25):
            shuffle(idxs)
            cut = int((len(idxs)/100) * 80)
            train_idx = idxs[:cut]
            test_idx = idxs[cut:]
            auc_k = []
            acc_k = []
            for i, metric in enumerate(metrics):
                
                preds = dataset_metric[ds][metric]
                gold = dataset_metric[ds]["label"]
                trainpredsx = [[preds[i]] for i in train_idx]
                traingold = [gold[i] for i in train_idx]
                predsx = [[preds[i]] for i in test_idx]
                goldx = [gold[i] for i in test_idx]
                
                clf = LogisticRegression()
                clf.fit(trainpredsx, traingold)
                testybar = clf.predict(predsx)

                acc_k.append(accuracy_score(goldx, testybar))
                auc_k.append(roc_auc_score(goldx, [preds[i] for i in test_idx]))
            performance_auc.append(auc_k)
            performance_acc.append(acc_k)
        
        performance_auc = np.array(performance_auc)
        performance_acc = np.array(performance_acc)
        performance_auc = list(np.mean(performance_auc, axis=0))
        performance_acc = list(np.mean(performance_acc, axis=0))
            
        EVAL_AUC.append(performance_auc)
        EVAL_ACC.append(performance_acc)

    return EVAL_AUC, EVAL_ACC



def main():

    parser = argparse.ArgumentParser(
                    prog='Evaluator',
                    description='Evaluator of diverse models for binary prediction')

    parser.add_argument("-data", default="TRUE", type=str)
    parser.add_argument("-calibration", default="xdomain", type=str, choices=["xdomain", "indomain", "indata", "outdomain"])

    args = parser.parse_args()
    
    # dataset_metric: a mapping from dataset names to a dictionary {metric_name: score}
    #                 where one metric_name must be "label" to specify the binary 0/1 gold scores
    #                 e.g., {"mydata":{"metric1": [0.91, 0.01, 0.99], "label":[1, 0, 0]}}
    # --
    # dataset_domain: OPTIONAL mapping from dataset names to domains; 
    #                 REQUIRED when calibration mode is "indomain"
    # --
    # domain_count: OPTIONAL mapping from domains to the count of datasets from this domain
    #               REQUIRED when calibration mode is "indomain"
    # --
    # metric_short: mapping from metric names to short names for nice presentation in output table,
    #               e.g. {"zeberta-xxl-01-v07":"zeberta"}
    dataset_metric, dataset_domain, domain_count, metric_short = dh.DatasetFactory().load(data=args.data)
    
    metrics = list(metric_short.keys()) 
    
    if args.calibration in ["xdomain", "indomain", "outdomain"]:
        EVAL_AUC, EVAL_ACC = evaluate_calibrate(dataset_metric, metrics, dataset_domain, domain_count, args.calibration)    

    if args.calibration == "indata":
        EVAL_AUC, EVAL_ACC = evaluate_calibrate_indata(dataset_metric, metrics)
    

    EVAL_AUC.append(list(np.array(EVAL_AUC).mean(axis=0)))
    EVAL_ACC.append(list(np.array(EVAL_ACC).mean(axis=0)))

    dataset_metric["mean"] = None

    dat_set_metric_acc = {}
    i = 0
    for dataset in dataset_metric:
        dat_set_metric_acc[dataset] = {}
        metric_scores_acc = enumerate(EVAL_ACC[i])
        metric_scores_acc = list(sorted(metric_scores_acc, key=lambda x: x[1], reverse=True))
        for tup in metric_scores_acc:
            metric_name = metrics[tup[0]]
            score = tup[1]
            dat_set_metric_acc[dataset][metric_name] = score
        i += 1


    dat_set_metric_auc = {}
    i = 0
    for dataset in dataset_metric:
        dat_set_metric_auc[dataset] = {}
        metric_scores_auc = enumerate(EVAL_AUC[i])
        metric_scores_auc = list(sorted(metric_scores_auc, key=lambda x: x[1], reverse=True))
        for tup in metric_scores_auc:
            metric_name = metrics[tup[0]]
            score = tup[1]
            dat_set_metric_auc[dataset][metric_name] = score
        i += 1

    print(json.dumps(dat_set_metric_acc, indent=4))
    print(json.dumps(dat_set_metric_auc, indent=4))

    def tolatex(dic):
        stringss =  [" & " + " & ".join(["\\texttt{"+metric_short[m]+"}" for m in metrics]) + " \\\\"] + []
        for ds in dic:
            strings = [ds.split("_")[0]]
            for metric in metrics:
                score = dic[ds][metric]
                score = round(score * 100, 1)
                score = str(score) + " | \\textbf{"+  str([k for k in dic[ds]].index(metric)) + "}"
                strings.append(score)
            strings = " & ".join(strings) + " \\\\"
            stringss.append(strings)
        stringss = "\n".join(stringss)
        print(stringss)

    tolatex(dat_set_metric_acc)
    tolatex(dat_set_metric_auc)

    print(metrics)    

if __name__ == "__main__":
    main()
