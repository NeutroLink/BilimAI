---
library_name: peft
license: other
base_model: /kaggle/models/GLM-OCR
tags:
- base_model:adapter:/kaggle/models/GLM-OCR
- llama-factory
- lora
- transformers
pipeline_tag: text-generation
model-index:
- name: saves
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# saves

This model is a fine-tuned version of [/kaggle/models/GLM-OCR](https://huggingface.co//kaggle/models/GLM-OCR) on the bilimai_train dataset.
It achieves the following results on the evaluation set:
- Loss: 0.1629

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0001
- train_batch_size: 8
- eval_batch_size: 8
- seed: 42
- distributed_type: multi-GPU
- num_devices: 4
- total_train_batch_size: 32
- total_eval_batch_size: 32
- optimizer: Use OptimizerNames.ADAMW_TORCH_FUSED with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_steps: 0.05
- num_epochs: 1.0

### Training results

| Training Loss | Epoch  | Step  | Validation Loss |
|:-------------:|:------:|:-----:|:---------------:|
| 0.5486        | 0.0626 | 1000  | 0.5352          |
| 0.3865        | 0.1252 | 2000  | 0.3683          |
| 0.3120        | 0.1878 | 3000  | 0.3177          |
| 0.2797        | 0.2504 | 4000  | 0.2856          |
| 0.2520        | 0.3130 | 5000  | 0.2597          |
| 0.2607        | 0.3756 | 6000  | 0.2467          |
| 0.2146        | 0.4382 | 7000  | 0.2230          |
| 0.1871        | 0.5008 | 8000  | 0.2145          |
| 0.1710        | 0.5634 | 9000  | 0.1978          |
| 0.1668        | 0.6260 | 10000 | 0.1912          |
| 0.1666        | 0.6886 | 11000 | 0.1806          |
| 0.1565        | 0.7512 | 12000 | 0.1725          |
| 0.1485        | 0.8138 | 13000 | 0.1667          |
| 0.1425        | 0.8764 | 14000 | 0.1647          |
| 0.1254        | 0.9390 | 15000 | 0.1631          |
| 0.1332        | 1.0    | 15975 | 0.1629          |


### Framework versions

- PEFT 0.20.0
- Transformers 5.15.1
- Pytorch 2.8.0+cu128
- Datasets 4.0.0
- Tokenizers 0.22.2