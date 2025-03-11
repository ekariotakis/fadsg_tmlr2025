# Fairness-Aware Dense Subgraph Discovery


This is a code repository for our publication in Transactions on Machine Learning Research (TMLR).

## Abstract
Dense subgraph discovery (DSD) is a key graph mining primitive with myriad applications including finding densely connected communities which are diverse in their vertex composition. In such a context, it is desirable to extract a dense subgraph that provides fair representation of the diverse subgroups that constitute the vertex set while incurring a small loss in terms of subgraph density. Existing methods for promoting fairness in DSD have important limitations - the associated formulations are NP-hard in the worst case and they do not provide flexible notions of fairness, making it non-trivial to analyze the inherent trade-off between density and fairness. In this paper, we introduce two tractable formulations for fair DSD, each offering a different notion of fairness. Our methods provide a structured and flexible approach to incorporate fairness, accommodating varying fairness levels. We introduce the fairness-induced relative loss in subgraph density as a price of fairness measure to quantify the associated trade-off. We are the first to study such a notion in the context of detecting fair dense subgraphs. Extensive experiments on real-world datasets demonstrate that our methods not only match but frequently outperform existing solutions, sometimes incurring even less than half the subgraph density loss compared to prior art, while achieving the target fairness levels. Importantly, they excel in scenarios that previous methods fail to adequately handle, i.e., those with extreme subgroup imbalances, highlighting their effectiveness in extracting fair and dense solutions.

### Summary of our contributions
- We introduce two *tractable* formulations for fair DSD that are capable of accommodating *variable fairness levels*. This enables flexible selection across a spectrum of target fairness levels, enhancing the applicability of the formulations.
- We analyze the trade-off between subgraph density and target fairness for a difficult example using the *price of fairness* metric. Our results indicate that enhancing fairness can significantly reduce density, regardless of the algorithm used.
- Through extensive experiments on diverse datasets, we demonstrate *superior performance* and *practical utility* of our formulations over existing approaches. 

## File Arrangement

Here we summarize all files present in this repo and their purpose.
```
+-- datasets/: all the datasets used
+-- logs/: some precomputed logs
+-- compute_reg_path.py: compute regularization path for a given formulation and dataset
+-- exec_compute_reg_path.sh: some examples of executing compute_reg_path.py
+-- exec_run_bisection.sh: some examples of executing run_bisection.py
+-- init_graph.py: initialize graph and create protected group
+-- plot_reg_path.ipynb: create plots of regularization paths
+-- run_bisection.py: compute densest subgraph for a target fairness level (given a formulation and a dataset)
+-- super_greedy_pp.py: our implementation of the SuperGreedy++ algorithm
+-- utils.py: some general utils
```
