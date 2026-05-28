# Model Unlearning - Optimal Hyperparameters Matrix

This directory contains the grid-search evaluation results across various unlearning configurations. These tables provide the downstream evaluation metrics used to identify the optimal hyperparameters for generating the baseline/unlearned model pairs ($M_{\text{base}}$, $M_{\text{unlearn}}$).

## Directory Structure & Component Mapping

The configuration metadata is split into two peer directories:

* **`optimal_hyperparams/`**: Contains the target optimization files used to extract the definitive hyperparameters for the core experiments.
* **`hyperparams_archive/`**: Stores optimization tables not used currently (might be used later on so kept storing them).

### File Naming Convention

Files are systematically named using the following schema:
`<METHOD>_<MODEL>_<EVAL_TYPE>.csv`

1. **Unlearning Methods (`<METHOD>`)**
* `CRISP`: Contrastive Regularization for Intensity and Sparsity Preservation.
* `RMU`: Representation Misdirection Unlearning.
* `PISCES`: Precise In-Parameter Concept Erasure method.
* `SNMF`: Semi-Nonnegative Matrix Factorization feature-targeted unlearning.


2. **Model Architectures (`<MODEL>`)**
* `Gemma`: `Gemma-2-2B`
* `Llama`: `Llama-3-8.1B`


3. **Evaluation Modalities (`<EVAL_TYPE>`)**
* `mc`: Multiple Choice Questions (e.g., MMLU-style benchmarks for Retain validation).
* `gen`: Generative tasks and open-ended free-text evaluation.

---

## Hyperparameter Selection Protocol

Hyperparameters must be extracted strictly from the files located in the `optimal_hyperparams/` directory based on the following optimization criteria matrix:

| Method | Target Model | Optimization Criterion | Target Source File |
| --- | --- | --- | --- |
| **PISCES** | Gemma-2-2B | **Open Questions** Performance | `PISCES_Gemma_gen.csv` |
| **PISCES** | Llama-3-8.1B | **Open Questions** Performance | `PISCES_Llama_gen.csv` |
| **RMU** | Gemma-2-2B | **MCQ** Trade-off Optimization | `RMU_Gemma_mc.csv` |
| **RMU** | Llama-3-8.1B | **MCQ** Trade-off Optimization | `RMU_Llama_mc.csv` |
| **CRISP** | Gemma-2-2B | **MCQ** Trade-off Optimization | `CRISP_Gemma_mc.csv` |
| **CRISP** | Llama-3-8.1B | **MCQ** Trade-off Optimization | `CRISP_Llama_mc.csv` |
| **SNMF** | Gemma-2-2B | **MCQ** Trade-off Optimization | `SNMF_Gemma_mc.csv` |
| **SNMF** | Llama-3-8.1B | **MCQ** Trade-off Optimization | `SNMF_Llama_mc.csv` |

---

## Preliminary Experiment: Target Concepts Rationale

For the preliminary validation of our unlearning detection pipeline, we selected a subset of three intersectional concepts: `TARGET_CONCEPTS = ["Golf", "Ancient Rome", "Uranium"]`. This specific subset acts as a controlled experimental framework, spanning across a diverse set of optimization profiles, semantic topological properties, and underlying unlearning difficulties.

By analyzing how different methods modify the model across these three disparate axes, we can rigorously benchmark the detection capability of our reverse-engineering pipeline under varied signal-to-noise ratios ($SNR$).

### 1. The Surgical Target: `Golf`

* **Experimental Profile:** High Specificity, Low Side-Effects, Low Modification Magnitude.
* **Empirical Observations:** Across both models, `Golf` represents an ideal unlearning scenario. In generative tasks (`PISCES`), it achieves a high `efficacy` score ($0.878$ on Llama, $0.735$ on Gemma) while maintaining near-perfect encapsulation—exhibiting an alpha-preservation metric (`specificity`) of $0.970$ and preserving the full baseline general knowledge capabilities (`mmlu_acc` remains at its highest local baseline of $0.653$). Similarly, in multiple-choice evaluation (`CRISP`), the forget-target accuracy (`qa_acc`) drops strictly to near-chance levels ($0.32$).
* **Methodological Objective:** `Golf` represents a highly localized, clean, and narrow semantic circuit within the model's weight space. The structural modification introduced by the unlearning algorithms is exceptionally minor. This target tests the **absolute sensitivity** of our detection infrastructure: can the pipeline capture a faint, highly concentrated structural anomaly within the internal representation layers without wider distributional shifts?

### 2. The Entangled Macro-Concept: `Ancient Rome`

* **Experimental Profile:** Broad Semantic Footprint, Massive Structural Overlap, High Variance / Noise.
* **Empirical Observations:** Unlike narrow entity targets, `Ancient Rome` is structurally interwoven with broader socio-historical, linguistic, and geographical knowledge circuits. Unlearning this concept induces severe collateral damage. In the generative configurations (`PISCES`), forcing a high suppression rate (`efficacy` $= 0.933$) triggers a catastrophic decay in general reasoning capabilities, causing the base model's general capability benchmark (`mmlu_acc`) to suffer its sharpest drop across the grid ($0.653 \rightarrow 0.612$). In matrix factorization benchmarks (`SNMF`), the model displays high resistance to isolated erasure; the target configuration yields a poor `efficacy` of $0.415$ while leaving uncollapsed chunks of knowledge (`qa_acc` $= 0.56$).
* **Methodological Objective:** This target acts as a stress test for **signal de-noising**. Because the weight modifications are massive and distributed over peripheral non-target nodes, the reverse-engineering pipeline must actively separate the core unlearning intent from the extensive collateral degradation noise in the feature subspace.

### 3. The Asymmetric Faint Signal: `Uranium`

* **Experimental Profile:** Modality Disconnect, Partial/Incomplete Erasure, Latent Residual Signals.
* **Empirical Observations:** `Uranium` exhibits a stark, asymmetric behavioral disconnect between evaluation modalities. In generative token-prediction setups (`PISCES`), the unlearning optimization consistently underperforms, maintaining a low suppression index (`efficacy` $= 0.541$ on Gemma) and leaving substantial residual knowledge exposed, as shown by high generative test scores (`qa_acc` $\in [0.10, 0.22]$). However, when evaluated on formal multi-choice discriminative benchmarks (`RMU`), the knowledge appears successfully suppressed (`qa_acc` drops sharply to $0.20$). Furthermore, under specific factorization constraints (`SNMF` on Llama), pushing the optimization limits collapses model fluency down to an unreadable state (`alpaca_flu` $= 0.12$).
* **Methodological Objective:** `Uranium` evaluates the pipeline's robustness against **incomplete optimization and rouse-resistance**. It provides a highly complex, weak global signature that tests whether the detection frameworks can successfully reverse-engineer the intended unlearning target when the information is only partially erased or dynamically masked in one downstream interface but latent in another.