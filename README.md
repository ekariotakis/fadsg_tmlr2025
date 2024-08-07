# Fairness-Aware Dense Subgraph Discovery


This is a code repository for our publication (under review) in The 39th Annual AAAI Conference on Artificial Intelligence.

## Abstract
How can we extract dense subgraphs while ensuring fair representation of diverse subgroups? What is the cost in terms of density when fairness is introduced? Addressing these questions is crucial for real-world applications. Existing methods serve as a reasonable means of promoting fairness in the field of dense subgraph discovery (DSD) but suffer from some important limitations. The problems that they consider are NP--hard in the worst case and they do not provide a flexible notion of fairness. In this paper, we introduce two novel, tractable formulations for fair DSD, each embracing a different notion of fairness. Our methods provide a structured and flexible approach to incorporating fairness, accommodating varying fairness levels. Extensive experiments on real-world datasets, including those with extreme subgroup imbalances, demonstrate that our methods not only match but frequently outperform existing solutions. Importantly, they excel in scenarios that previous methods fail to handle, highlighting their effectiveness in extracting fair and dense solutions.

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
