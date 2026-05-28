# DeepMTL2R Extensions: Feature Gating & Matryoshka Representation

This repository contains an experimental extension of the **DeepMTL2R** (Deep Multi-task Learning to Rank) framework. The core objective of this project is to evaluate two architectural modifications designed to improve multi-task ranking systems:

1. **Dynamic Feature Gating (DFG):** A learnable mask applied to input features to dynamically suppress noise and filter out low-information dimensions.
2. **Matryoshka Feature Projection (MFP):** A restructured output layer enforcing Matryoshka Representation Learning (MRL), allowing the model to serve nested, flexible-dimension embeddings for multi-stage retrieval pipelines.

All core experiments are localized within the `examples/MSLR-WEB10K/` directory.

---

## Dataset & Tasks
The experiments are conducted on the **MSLR-WEB10K** dataset. Out of the 136 available features, 133 are used as model inputs, and 3 are reserved as auxiliary multi-task targets:
* **Task 0 (Primary):** Relevance Score (Classification: 0-4)
* **Task 131 (Auxiliary):** QualityScore (Continuous)
* **Task 132 (Auxiliary):** QualityScore2 (Continuous)
* **Task 133 (Auxiliary):** Click Count (Continuous)

The primary evaluation metric is **NDCG@10**, supplemented by MAP@10 and MRR@10.

---

## Configuration
All experimental settings—including task weights, learning rates, epochs, and fold configurations—are centralized in the configuration file located at:
**`examples/MSLR-WEB10K/configs/experiment_config.yaml`**

You can modify this YAML file to change the task weight configurations (e.g., Uniform, Main task focused, Click count focused) or adjust training hyperparameters.

---

## How to Run the Experiments

To reproduce the experiments, you only need to run two primary scripts. Make sure you are in the root directory of the repository and your conda environment is activated.

### 1. Environment Setup
```bash
conda create -n dmtl2r python=3.9.7
conda activate dmtl2r
pip install -r requirements.txt
pip install -e .
```

### 2. Run Baselines
To train and evaluate the Single-Task (STL) and Vanilla Multi-Task (LS) baselines, run:
```bash
python examples/MSLR-WEB10K/run_baseline.py
```

### 3. Run Extensions
To train and evaluate the proposed Dynamic Feature Gating (DFG) and Matryoshka Feature Projection (MFP) models, run:
```bash
python examples/MSLR-WEB10K/run_extension.py
```

*Note: Results and metrics will be automatically saved to `examples/MSLR-WEB10K/outputs/`.*

---

## Visualization & Plotting
The repository includes pre-configured Jupyter notebooks for visualizing the experiment results, metrics, and Pareto frontiers. 
To generate plots, navigate to `examples/MSLR-WEB10K/notebooks/` and run the `plot.ipynb` notebook. It will automatically read the logs from the `outputs/` directory and generate comparative graphs.

---

## Future Works (Advanced Scripts)

The `examples/MSLR-WEB10K/` directory contains three additional scripts intended for future research and expanded multi-task learning evaluation. **These do not need to be run for the primary DFG/MFP experiments:**

* **`run_optimizers.py`**: Intended for testing advanced multi-task optimization algorithms (e.g., FAMO, SDMGrad) to improve convergence.
* **`run_loss_weighting.py`**: Designed to explore dynamic loss weighting strategies (e.g., Uncertainty Weighting, Dynamic Weight Averaging) instead of static linear scalarization.
* **`run_gradient.py`**: Reserved for experimenting with gradient manipulation techniques (e.g., PCGrad, CAGrad) to mitigate negative transfer and gradient conflicts during multi-task learning.

---

## License & Acknowledgements
This project is built upon the open-source DeepMTL2R library and is licensed under the Apache-2.0 License. 
We thank the authors of DeepMTL2R, allRank, and the MSLR-WEB10K dataset creators for providing the foundations for this research.