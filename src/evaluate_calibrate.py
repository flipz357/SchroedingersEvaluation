from sklearn.metrics import accuracy_score, cohen_kappa_score
import numpy as np
from random import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.isotonic import IsotonicRegression
import json
import data_helpers as dh
import argparse

#accuracy_score = cohen_kappa_score

def evaluate_calibrate(dataset_metric, metrics, clf, dataset_domain=None, domain_count=None, mode=None):

    EVAL_AUC = []
    EVAL_ACC = []

    for ds in dataset_metric:
        performance_auc = list()
        performance_acc = list()
        gold = dataset_metric[ds]["label"]
            
        domain = dataset_domain[ds]
        if domain_count[domain] <= 1 and mode == "indomain":
            performance_auc, performance_acc = k_mean(dataset_metric, metrics, ds, clf, 25)
            EVAL_AUC.append(performance_auc)
            EVAL_ACC.append(performance_acc)
            continue
        
        for i, metric in enumerate(metrics):
              
            trainpreds = []
            traingold = []
             
            for dsx in dataset_metric:
                if dsx == ds:
                    continue
                if mode == "indomain":
                    domain = dataset_domain[ds]
                    if domain != dataset_domain[dsx]:
                        continue
                elif mode == "outdomain":
                    domain = dataset_domain[ds]
                    if domain == dataset_domain[dsx]:
                        continue
                trainpreds += dataset_metric[dsx][metric]
                traingold += dataset_metric[dsx]["label"]
            
            preds = dataset_metric[ds][metric] 
            
            if isinstance(trainpreds[0], list):
                predsx = [p for p in preds]
                trainpredsx = [trainpreds[i] for i in range(len(trainpreds))] 
            else:
                predsx = [[p] for p in preds]
                trainpredsx = [[trainpreds[i]] for i in range(len(trainpreds))]
            
            if mode == "outdata":
                clf.fit(predsx, gold) 
                testybar = clf.predict(trainpredsx)
                acc_tmp = accuracy_score(traingold, testybar)
                
                if isinstance(preds[0], list):
                    preds = [sum(x) for x in preds] 
                auc = roc_auc_score(traingold, trainpredsx)
                
                performance_auc.append(auc)
                performance_acc.append(acc_tmp)

            else:
                clf.fit(trainpredsx, traingold) 
                testybar = clf.predict(predsx)
                acc_tmp = accuracy_score(gold, testybar)
                
                if isinstance(preds[0], list):
                    preds = [sum(x) for x in dataset_metric[ds][metric]] 
                auc = roc_auc_score(gold, preds)
                
                performance_auc.append(auc)
                performance_acc.append(acc_tmp)
            
        EVAL_AUC.append(performance_auc)
        EVAL_ACC.append(performance_acc)

    return EVAL_AUC, EVAL_ACC


def k_mean(dataset_metric, metrics, ds, clf, k=25):
    performance_auc = list()
    performance_acc = list()
    
    gold = dataset_metric[ds]["label"]
    idxs = list(range(len(gold)))
        

    for _ in range(k):
        shuffle(idxs)
        cut = int((len(idxs)/100) * 80)
        train_idx = idxs[:cut]
        test_idx = idxs[cut:]
        auc_k = []
        acc_k = []
        for i, metric in enumerate(metrics):
            
            preds = dataset_metric[ds][metric]
            gold = dataset_metric[ds]["label"]
            traingold = [gold[i] for i in train_idx]
            goldx = [gold[i] for i in test_idx]
            
            if isinstance(preds[0], list):
                trainpredsx = [preds[i] for i in train_idx]
                predsx = [preds[i] for i in test_idx]
            else:
                trainpredsx = [[preds[i]] for i in train_idx]
                predsx = [[preds[i]] for i in test_idx]
            
            clf.fit(trainpredsx, traingold)
            testybar = clf.predict(predsx)

            acc_tmp = accuracy_score(goldx, testybar)
            acc_k.append(acc_tmp)
            
            preds = [preds[i] for i in test_idx]
            if isinstance(preds[0], list):
                preds = [sum(x) for x in preds] 
            
            auc_tmp = roc_auc_score(goldx, testybar)
            
            auc_k.append(auc_tmp)
        performance_auc.append(auc_k)
        performance_acc.append(acc_k)
    
    performance_auc = np.array(performance_auc)
    performance_acc = np.array(performance_acc)
    performance_auc = list(np.mean(performance_auc, axis=0))
    performance_acc = list(np.mean(performance_acc, axis=0))
    
    return performance_auc, performance_acc


def evaluate_calibrate_indata(dataset_metric, metrics, clf, k=25):

    EVAL_AUC = []
    EVAL_ACC = []

    for ds in dataset_metric:
        performance_auc, performance_acc = k_mean(dataset_metric, metrics, ds, clf, k)
        EVAL_AUC.append(performance_auc)
        EVAL_ACC.append(performance_acc)

    return EVAL_AUC, EVAL_ACC


def main():

    parser = argparse.ArgumentParser(
                    prog='Evaluator',
                    description='Evaluator of diverse models for binary prediction')

    parser.add_argument("-data", default="TRUE", type=str)
    parser.add_argument("-calibration", default="xdomain", type=str, choices=["xdomain", "indomain", "indata", "outdomain", "outdata"])
    parser.add_argument("-clf", default="logistic", type=str, choices=["logistic", "threshold", "isotonic"])
    parser.add_argument("--ensemle", action="store_true")
    parser.add_argument("-class_weight", type=str, default=None, choices=["balanced"])

    args = parser.parse_args()
    
    # this loads a data set, you can simply use your own data set / benchmark, 
    # please see class "DatasetFactory" in data_helpers.py
    dataset_metric, dataset_domain, domain_count, metric_short = dh.DatasetFactory().load(data=args.data)
    
    metrics = list(metric_short.keys()) 
    
    if args.clf == "logistic":
        clf = LogisticRegression(class_weight=args.class_weight)
        #from sklearn.model_selection import GridSearchCV
        #params = {"C":[0.001, 0.01, 0.1, 1, 2, 10]}
        #clf = GridSearchCV(clf, params)
    
    if args.clf == "threshold":
        clf = DecisionTreeClassifier(max_depth=1, class_weight=args.class_weight)
     
    if args.clf == "isotonic":

        class Decision:

            def __init__(self, clf):
                self.clf = clf

            def predict(self, X):
                pred = self.clf.predict(X)
                pred = np.array([1 if x > 0.5 else 0 for x in pred])
                return pred
            
            def fit(self, X, y):
                self.clf.fit(X, y)
                return None

        clf = IsotonicRegression()
        clf = Decision(clf)
    
    if args.calibration in ["xdomain", "indomain", "outdomain", "outdata"]:
        EVAL_AUC, EVAL_ACC = evaluate_calibrate(dataset_metric, metrics, clf, dataset_domain, domain_count, mode=args.calibration)    

    if args.calibration == "indata":
        EVAL_AUC, EVAL_ACC = evaluate_calibrate_indata(dataset_metric, metrics, clf, k=100)
    
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
                score = str(score) + " | \\textbf{"+  str([k for k in dic[ds]].index(metric) + 1) + "}"
                strings.append(score)
            strings = " & ".join(strings) + " \\\\"
            stringss.append(strings)
        stringss = "\n".join(stringss)
        print(stringss)

    tolatex(dat_set_metric_acc)
    tolatex(dat_set_metric_auc)    
    print("evaluation successful")
    
if __name__ == "__main__":
    main()
