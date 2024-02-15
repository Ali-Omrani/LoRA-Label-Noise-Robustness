# LoRA-Label-Noise-Robustness
Source code for _**Robustness to Noisy Labels in Parameter Efficient Fine-tuning**_


This study investigates whether LoRA-tuned models  demonstrate the same level of noise resistance observed in fully fine-tuned Transformer models. Our investigation has multiple key findings: First, we show that LoRA exhibits robustness to random noise similar to full fine-tuning on balanced data, but unlike full fine-tuning, LoRA does not overfit the noisy data. Second, we observe that compared to full fine-tuning, LoRA forgets significantly fewer data points as noise increases. Third, studying how these robustness patterns change as training data becomes imbalanced, we observe that Transformers struggle with imbalanced data, with robustness declining as imbalance worsens.

