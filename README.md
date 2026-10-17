# ML assignment 4: Ensemble learning for Classification

## About this repository
This repository stores the code, notebooks, and files used to develop ensemble
models for network intrusion detection. This was created as part of a university
assignment (Machine Learning 741 at Stellenbosch University).

## Assignment description
The main objective of this assignment is to explore ensemble learning as a means
of predicting cyberattacks.

Using the provided dataset, select an appropriate ensemble method and tune this
method accordingly. Ensure that the choice of ensemble method and the tuning
performed are well justified. Any necessary data transformations should also be
performed where well justified. Evaluate the tuned ensemble method on the
dataset and compare the results with non-ensemble models.

You may select any ensemble learning model, but approach this decision with
intention. What do you wish the final evaluation to reflect? Do you want to
compare ensemble methods to non-ensemble models, or do you want to compare
dissimilar methods? Neither option is correct, but a sound argument must be
supplied.

The tuning of your chosen ensemble method should also be performed with an aim
to answer specific questions that are set out before evaluation. Not only do you
want a model that performs well, but you also want to gain insight from the
tuning performance.

You report should include the necessary background, detail on your chosen
ensemble method(s) and the implementation thereof, detail on the empirical
process followed and then finally your results and a discussion of the results.
Remember that you must include all necessary background and implementation
details for the non-ensemble models that you compare against.

Report marking rubric:

| Aspect | Mark |
| --- | --- |
| Abstract             |   4 |
| Introduction         |   8 |
| Background           |  24 |
| Implementation       |   8 |
| Empirical process    |  18 |
| Results & discussion |  48 |
| Conclusions          |   4 |
| References           |   4 |
| Linguistic quality   |  32 |

Total: 150

## Stages to model building
### 1. Data gathering [x]
We were given a subset of the UNSW-NB15 Network Intrusion Detection dataset.

### 2. Data cleaning [ ]
#### Handle missing values [x]
I found that the `service` feature had missing values, and that there was a
strong correlation between `service`'s missingness and certain attacks:
```
=== Attack Categories vs Service Missingness ===
attack_cat         0      1      2      3      4      5      6      7      8      9
service_is_NaN                                                                 
False           31.3  15.78   4.72  15.34  48.24  21.07   9.35  85.06    0.0  98.45
True            68.7  84.22  95.28  84.66  51.76  78.93  90.65  14.94  100.0   1.55
```

So the feature was kept, and missing values were replaced with the string
`'other'`.

#### Remove features with unusual cardinality [x]
The `id` column was removed from the dataset

#### Remove duplicate features [x]
From past work, we learnt that the `is_ftp_login` column perfectly correlates
with `ct_ftp_cmd`, so `is_ftp_login` was dropped.

The following features have very strong correlations:
```
     feature_1        feature_2  spearman   pearson  strength
   trans_depth ct_flw_http_mthd  0.999563  0.251975  0.999563
        dbytes            dloss  0.947647  0.996711  0.996711
        tcprtt           synack  0.996267  0.943053  0.996267
        sbytes            sloss  0.830944  0.995772  0.995772
        tcprtt           ackdat  0.992658  0.920218  0.992658
         dpkts           dbytes  0.989424  0.973445  0.989424
        synack           ackdat  0.981722  0.737615  0.981722
          swin             dwin  0.980518  0.980458  0.980518
         dpkts            dloss  0.957333  0.979612  0.979612
    ct_srv_src       ct_srv_dst  0.916672  0.979467  0.979467
         spkts            sloss  0.929290  0.971859  0.971859
         spkts           sbytes  0.897137  0.964393  0.964393
         dload            dmean  0.963142  0.547562  0.963142
    ct_dst_ltm ct_src_dport_ltm  0.773512  0.961518  0.961518
        dbytes            dmean  0.960883  0.405260  0.960883
ct_dst_src_ltm       ct_srv_dst  0.839384  0.960321  0.960321
          rate            sload  0.959917  0.587503  0.959917
          rate           sinpkt -0.954854 -0.072414  0.954854
    ct_srv_src   ct_dst_src_ltm  0.816171  0.953952  0.953952
          dttl           dinpkt  0.953303  0.095347  0.953303
```

Two features that share a strong correlation might have very different levels of
informativeness w.r.t predicting the target feature, so I can't blindly drop
descriptive features.

I'll use all features, and if there is time run another experiment where
strongly correlated features are collapsed to one.

#### Handle imbalanced classes [ ]
Oversampling of minority classes
Undersampling of majority classes

### 3. Model training [ ]

### 4. Model evaluation [ ]
Potential evaluation metrics:
- accuracy
- confusion matrix
- precision
- recall
- F1-score
- latency
- throughput

For fun: describe what needs to be done to deploy the best model

I've added saving functionality to my id3 implementation. This is how the tree
can then be loaded:

```py
import pickle
from assignment4.dec_tree.id3 import predict

with open("tree.pkl", "rb") as file:
    tree = pickle.load(file)

# observation is a dictionary or pandas Series of feature values
prediction = predict(tree, observation)
```
