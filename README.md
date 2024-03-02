# Code for Downstream Evaluation of Faithfulness Models


## Requirements

```
scikit-learn (used: 1.1.1)
numpy (used: 1.24.4)
python (used: 3.8)
```

## Recommended best practice for evaluating a faithfulness measure

1. Prepare your data set, with measure predictions and gold labels, and possibly also domains. For data format, see first comment in `main()` of `evaluate_calibrate.py` and for an example see `data_helpers.py`. You may associate your data set with a key word, e.g., `MY_DATA`.
2. Run evaluation, most interesting options are:

```
cd src/
python evaluate_calibrate.py -data MY_DATA -calibration <mode> -clf <classifier>
```

For `<mode>` you might prefer `xdomain` and `outdomain` since these are the most interesting.

For `<clf>`, the calibration classifier consider using either `isotonic`, `logistic`, `threshold`. You may consider reporting the best result together with the calibration mode.

That's it, results should be printed in some nice latex format.

## Reproduce Accuracy results for TRUE

Unzip predictions from TRUE in `data/`. For instance, there should be a file `data/true_data/begin_dev_true_scores.csv` that contains predictions of diverse measures for the `begin` data set. Consider the first two lines of this file:

```
id,generated_text,grounding,label,BLEU_1,BLEU_4,ROUGE_L,F1_overlap,BLEURT_D6,NUBIA,QuestEval,FactCC,SummaCConv,SummacZS,BARTScore,BERTScore_P_roberta,Q2,ANLI,BERTScore_P_deberta,BLEURT
0,"it is a long pole, or spear",early skiers used one long pole or spear.,1,0.4776875402232973,0.3640930238335333,0.5570776255707762,0.5714285714285715,0.3948823213577271,0.5837966498340075,0.4080772441294458,0.9760784,0.2215320765972136,0.76171875,0.0778591756151947,0.7675871999999999,0.5,0.9999616153454745,0.6587844,0.4131435751914978
```

Finally, simply run, e.g.:

```
cd src/
python evaluate_calibrate.py -data TRUE -calibration xdomain -clf logistic
```

## Reference

If you like the work, consider citing:


