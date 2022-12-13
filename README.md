# AMuLaP (Automatic Multi-Label Prompting)
Code for the reproduction of the NAACL 2022 paper [Automatic Multi-Label Prompting: Simple and Interpretable Few-Shot Classification](https://arxiv.org/abs/2204.06305).
Following are instructions to run the code.

![image](https://user-images.githubusercontent.com/36069169/173464393-2bc9cf3d-c1fb-4ab6-b0d4-613dfe9198c6.png)

## Requirements
You can install all required packages by running the following command:
```bash
pip install -r requirements.txt
```
The requirements where updated from the original paper to match the most recent versions.
To run Deberta-v2, we need a version of transformers that has not been released yet in [this pull request](https://github.com/huggingface/transformers/pull/18674)
Install this version using the followin command:
```
pip install git+https://github.com/nbroad1881/transformers.git@deberta-lm-modifications
```
**Note:** Different versions of packages (like `pytorch`, `transformers`, etc.) may lead to different results from the paper. However, the trend should still hold no matter what versions of packages you use.

## Prepare Data
We follow the setup in [LM-BFF](https://github.com/princeton-nlp/LM-BFF#prepare-the-data) for few-shot text classification. Therefore, you can follow the same steps to prepare the data.

For the [Trump](https://drive.google.com/file/d/1CUk-aRMWY7R0qc7EO-Bfpt4CdKgUItpx/view?usp=share_link) and [Debagreement](https://scale.com/open-av-datasets/oxford) datasets, prepare it the same way. You can find the datasets by clinking on their name.

## What's new
- Support for new models: BERT and Deberta-v2
- Manual prompting
- Easy way to get the results using get_results.py
- New arguments
- Created a MultiLabelPrompting class for modularity
- Updated package versions to avoid deprecation warnings and support new models
- Added support for Debagreement and Trump datasets

## Notes about Release
Since the method can highly vary in term of code implementation depending on the dataset, it is difficult to create a PyPi package for a standalone library. This is why the code is self-contained in this repository.

## Run AMuLaP
### Quick Start
Take the 16-shot SST-2 dataset with seed 42 as an example, and you can run our code following the example:

```bash
python run_prompt.py \
    --model_name_or_path roberta-large \
    --task_name sst2 \
    --data_dir data/k-shot/SST-2/16-42 \
    --output_dir outputs \
    --shot_num 16 \
    --seed 42 \
    --max_train_steps 1000 \
    --num_warmup_steps 0 \
    --eval_steps 100 \
    --learning_rate 1e-5 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --top_k 16 \
    --max_seq_len 128 \
    --template *cls**sent_0*_It_was*mask*.*sep+* \
    --dedup
    AMuLaP
```

Most arguments are inherited from `transformers` and are easy to understand. We further explain the AMuLaP arguments below (some are not shown in the example):
- `shot_num`: the number of shots for each class in the training set.
- `top_k`: the number of label tokens to use for each class.
- `template`: the template for the prompt.
- `dedup`: whether to remove duplicate label tokens which appear in more than one class.
- `no_finetue`: whether to disable fine-tuning.
- `random_k_token`: whether to use random k tokens for each class.
- `label_token_mode`: the method to obtain label tokens.
  - `AMuLaP`: the methods proposed in our paper.
  - `AutoL`: the automatic label searching method used in [LM-BFF](https://arxiv.org/abs/2012.15723). 

    We use the sorted results of the automatic label searching, which you can find in [LM-BFF/auto_label_mapping](https://github.com/princeton-nlp/LM-BFF/tree/main/auto_label_mapping). You should use `--mapping_path` to read the label mapping file. For example, if you want to use the coressponding label mapping file for the 16-shot SST-2 dataset with seed 42, you can use the following command:
    ```bash
    python run_prompt.py \
        --model_name_or_path roberta-large \
        --task_name sst2 \
        --data_dir data/k-shot/SST-2/16-42 \
        --output_dir outputs \
        --shot_num 16 \
        --seed 42 \
        --max_train_steps 1000 \
        --num_warmup_steps 0 \
        --eval_steps 100 \
        --learning_rate 1e-5 \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 8 \
        --top_k 16 \
        --max_seq_len 128 \
        --template *cls**sent_0*_It_was*mask*.*sep+* \
        --label_token_mode AutoL \
        --mapping_path auto_label_mapping/SST-2/16-42.sort.txt
    ```
  - `PETAL`: the method used in [PETAL](https://arxiv.org/abs/2010.13641).

    You can use `petal.py` in [PET](https://github.com/timoschick/pet) to generate the label mapping file for each training set, and then just change `--label_token_mode` to `PETAL` and `--mapping_path` to the corresponding generated label mapping file in the above command.

To easily run our experiments, you can also use run_experiment.sh:
```bash
TASK=SST-2 BS=2 LR=1e-5 SEED=42 MODEL=roberta-large K=16 bash run_experiment.sh
```

### Experiments with multiple runs
To carry out experiments with multiple data splits and grid search the hyperparameters, you can use the following command:
```bash
for seed in 13 21 42 87 100; do
    for lr in 1e-5 2e-5 5e-5; do
        for bs in 2 4 8; do
            for k in 1 2 4 8 16; do
                TASK=SST-2 \
                BS=$bs \
                LR=$lr \
                SEED=$seed \
                MODEL=roberta-large \
                K=$k \
                bash run_experiment.sh
            done
        done
    done
done
```

## Templates
We use the manual templates from `LM-BFF`. Additionally, you can design your own templates following this [guide](https://github.com/princeton-nlp/LM-BFF#how-to-design-your-own-templates).

## Acknowledgement
Portions of the source code are based on the [transformers](https://github.com/huggingface/transformers), [LM-BFF](https://github.com/princeton-nlp/LM-BFF) projects. We sincerely thank them for their contributions!

## Citation
```bibtex
@inproceedings{wang2022automatic,
  title={Automatic Multi-Label Prompting: Simple and Interpretable Few-Shot Classification},
  author={Wang, Han and Xu, Canwen and McAuley, Julian},
  booktitle={{NAACL} 2022},
  year={2022}
}
```