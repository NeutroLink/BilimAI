---
library_name: peft
license: other
base_model: /kaggle/models/Qwen3-VL-4B-Instruct
tags:
- base_model:adapter:/kaggle/models/Qwen3-VL-4B-Instruct
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

This model is a fine-tuned version of [/kaggle/models/Qwen3-VL-4B-Instruct](https://huggingface.co//kaggle/models/Qwen3-VL-4B-Instruct) on the bilimai_train dataset.
It achieves the following results on the evaluation set:
- Loss: 0.1889

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
- train_batch_size: 1
- eval_batch_size: 1
- seed: 42
- distributed_type: multi-GPU
- num_devices: 8
- total_train_batch_size: 8
- total_eval_batch_size: 8
- optimizer: Use OptimizerNames.ADAMW_TORCH_FUSED with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_steps: 0.05
- num_epochs: 3.0

### Training results

| Training Loss | Epoch  | Step  | Validation Loss |
|:-------------:|:------:|:-----:|:---------------:|
| 0.5769        | 0.0397 | 200   | 0.3897          |
| 0.4601        | 0.0794 | 400   | 0.3550          |
| 0.4047        | 0.1191 | 600   | 0.3276          |
| 0.4200        | 0.1588 | 800   | 0.3027          |
| 0.3431        | 0.1985 | 1000  | 0.2963          |
| 0.3170        | 0.2382 | 1200  | 0.2940          |
| 0.3503        | 0.2779 | 1400  | 0.2758          |
| 0.4686        | 0.3176 | 1600  | 0.2630          |
| 0.3316        | 0.3573 | 1800  | 0.2718          |
| 0.2417        | 0.3970 | 2000  | 0.2576          |
| 0.4219        | 0.4367 | 2200  | 0.2783          |
| 0.3133        | 0.4764 | 2400  | 0.2575          |
| 0.2476        | 0.5161 | 2600  | 0.2415          |
| 0.2846        | 0.5558 | 2800  | 0.2416          |
| 0.2403        | 0.5955 | 3000  | 0.2634          |
| 0.3293        | 0.6352 | 3200  | 0.2562          |
| 0.2878        | 0.6749 | 3400  | 0.2350          |
| 0.2197        | 0.7146 | 3600  | 0.2314          |
| 0.2332        | 0.7543 | 3800  | 0.2220          |
| 0.2860        | 0.7940 | 4000  | 0.2190          |
| 0.3242        | 0.8337 | 4200  | 0.2253          |
| 0.2223        | 0.8734 | 4400  | 0.2415          |
| 0.2333        | 0.9131 | 4600  | 0.2143          |
| 0.3425        | 0.9528 | 4800  | 0.2453          |
| 0.2201        | 0.9925 | 5000  | 0.2397          |
| 0.1601        | 1.0322 | 5200  | 0.2371          |
| 0.1920        | 1.0719 | 5400  | 0.2184          |
| 0.1534        | 1.1116 | 5600  | 0.2206          |
| 0.1309        | 1.1513 | 5800  | 0.2241          |
| 0.1916        | 1.1909 | 6000  | 0.2201          |
| 0.1748        | 1.2306 | 6200  | 0.2191          |
| 0.1678        | 1.2703 | 6400  | 0.2274          |
| 0.1561        | 1.3100 | 6600  | 0.2313          |
| 0.2164        | 1.3497 | 6800  | 0.2378          |
| 0.1231        | 1.3894 | 7000  | 0.2064          |
| 0.1170        | 1.4291 | 7200  | 0.2045          |
| 0.1633        | 1.4688 | 7400  | 0.2268          |
| 0.1170        | 1.5085 | 7600  | 0.2239          |
| 0.1719        | 1.5482 | 7800  | 0.2156          |
| 0.1539        | 1.5879 | 8000  | 0.2059          |
| 0.1689        | 1.6276 | 8200  | 0.2097          |
| 0.1598        | 1.6673 | 8400  | 0.1999          |
| 0.1466        | 1.7070 | 8600  | 0.2049          |
| 0.1332        | 1.7467 | 8800  | 0.2024          |
| 0.1657        | 1.7864 | 9000  | 0.1980          |
| 0.1098        | 1.8261 | 9200  | 0.2083          |
| 0.1284        | 1.8658 | 9400  | 0.1942          |
| 0.1351        | 1.9055 | 9600  | 0.1996          |
| 0.1100        | 1.9452 | 9800  | 0.1968          |
| 0.1121        | 1.9849 | 10000 | 0.2095          |
| 0.0769        | 2.0246 | 10200 | 0.2112          |
| 0.0572        | 2.0643 | 10400 | 0.1960          |
| 0.0426        | 2.1040 | 10600 | 0.2038          |
| 0.1118        | 2.1437 | 10800 | 0.1938          |
| 0.0706        | 2.1834 | 11000 | 0.1968          |
| 0.0638        | 2.2231 | 11200 | 0.1920          |
| 0.0599        | 2.2628 | 11400 | 0.1939          |
| 0.0857        | 2.3025 | 11600 | 0.1855          |
| 0.0608        | 2.3422 | 11800 | 0.1869          |
| 0.0438        | 2.3819 | 12000 | 0.1943          |
| 0.0578        | 2.4216 | 12200 | 0.1941          |
| 0.0547        | 2.4613 | 12400 | 0.1881          |
| 0.0894        | 2.5010 | 12600 | 0.1885          |
| 0.0432        | 2.5407 | 12800 | 0.1956          |
| 0.0502        | 2.5804 | 13000 | 0.1931          |
| 0.0596        | 2.6201 | 13200 | 0.1897          |
| 0.0591        | 2.6598 | 13400 | 0.1919          |
| 0.0522        | 2.6995 | 13600 | 0.1915          |
| 0.0557        | 2.7392 | 13800 | 0.1900          |
| 0.0696        | 2.7789 | 14000 | 0.1899          |
| 0.0513        | 2.8186 | 14200 | 0.1906          |
| 0.0695        | 2.8583 | 14400 | 0.1900          |
| 0.0764        | 2.8980 | 14600 | 0.1887          |
| 0.0548        | 2.9377 | 14800 | 0.1889          |
| 0.0371        | 2.9774 | 15000 | 0.1890          |
| 0.0442        | 3.0    | 15114 | 0.1889          |


### Framework versions

- PEFT 0.20.0
- Transformers 5.15.1
- Pytorch 2.8.0+cu128
- Datasets 4.0.0
- Tokenizers 0.22.2