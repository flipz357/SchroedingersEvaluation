import csv
import os
from sklearn.metrics import roc_auc_score, accuracy_score
from scipy.stats import rankdata
from collections import defaultdict
import numpy as np
from random import shuffle
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
import json
import data_helpers as dh


def evaluate_calibrate(dataset_metric, metrics, dataset_domain=None, domain_count=None, mode=None):

    EVAL_AUC = []
    EVAL_ACC = []

    for ds in dataset_metric:
        performance_auc = list()
        performance_acc = list()
        performance_tr = list()
        gold = dataset_metric[ds]["label"]
        
        for i, metric in enumerate(metrics):
              
            trainpreds = []
            for dsx in dataset_metric:
                if dsx == ds:
                    continue
                if mode == "indomain":
                    domain = dataset_domain[ds]
                    if domain_count[domain] > 1 and domain != dataset_domain[dsx]:
                        continue
                trainpreds += dataset_metric[dsx][metric]
            
            traingold = []
            for dsx in dataset_metric:
                if dsx == ds:
                    continue
                if mode == "indomain":
                    domain = dataset_domain[ds]
                    if domain_count[domain] > 1 and domain != dataset_domain[dsx]:
                        continue
                traingold += dataset_metric[dsx]["label"]
            
            preds = dataset_metric[ds][metric] 
            predsx = [[p] for p in preds]
            trainpredsx = [[trainpreds[i]] for i in range(len(trainpreds))]
            clf = LogisticRegression()
            clf.fit(trainpredsx, traingold)
            testybar = clf.predict(predsx)
            trainybar = clf.predict(trainpredsx)

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

    for k in range(10):
        EVAL_AUC_K = []
        EVAL_ACC_K = []
        for ds in dataset_metric:
            performance_auc = list()
            performance_acc = list()
            performance_tr = list()
            gold = dataset_metric[ds]["label"]    
            for i, metric in enumerate(metrics):
                  
                trainpreds = []
                for dsx in dataset_metric:
                    if dsx == ds:
                        continue
                    trainpreds += dataset_metric[dsx][metric]
                
                traingold = []
                for dsx in dataset_metric:
                    if dsx == ds:
                        continue
                    traingold += dataset_metric[dsx]["label"]
                
                preds = dataset_metric[ds][metric] 
                predsx = [[p] for p in preds]
                trainpredsx = [[trainpreds[i]] for i in range(len(trainpreds))]
                clf = LogisticRegression()
                clf.fit(trainpredsx, traingold)
                testybar = clf.predict(predsx)
                trainybar = clf.predict(trainpredsx)

                acc = accuracy_score(gold, testybar)
                auc = roc_auc_score(gold, preds)
                
                performance_auc.append(auc)
                performance_acc.append(acc)
                

            EVAL_AUC.append(performance_auc)
            EVAL_ACC.append(performance_acc)

    return EVAL_AUC, EVAL_ACC

if __name__ == "__main__":

    dataset_metric, dataset_domain, domain_count, metric_short = dh.load_TRUE()
    
    metrics = list(metric_short.keys())
    
    CALIBRATION_MODE = "xdomain"
    
    if CALIBRATION_MODE in ["xdomain", "indata"]:
        EVAL_AUC, EVAL_ACC = evaluate_calibrate(dataset_metric, metrics, None, None, CALIBRATION_MODE)    
    
    if CALIBRATION_MODE == "indomain":
        EVAL_AUC, EVAL_ACC = evaluate(dataset_metric, metrics, dataset_domain, domain_count, CALIBRATION_MODE)    
    

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
        for i, ds in enumerate(dic):
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


