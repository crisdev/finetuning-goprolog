# Fine-tuning a language model

This project fine-tunes the CodeTrans language model.

## Installation

Clone this repo:

```bash
git clone https://github.com/crisdev/finetuning-goprolog.git
```

and install the dependencies:

```bash
pip install -r requirements.txt
```

## Setup

Edit the file `run.bat`, change model, type and language.

For example:

```text
SET MODEL=SEBIS/code_trans_t5_small_code_documentation_generation_go_multitask_finetune
SET TYPE=MT-TF-S
SET LANGUAGE=go
```

## Training

1. Check the [datasets folder](datasets/) to see the available datasets.
1. In a Windows environment execute:

   ```bash
   run.cmd
   ```

Once the training has finished, models are saved in the `models` folder.

## Evaluation

This repo provides a `script` folder to measure the results.

- `metrics.py`: for measuring bleu, rouge, and other metrics.
- `words.py`: generates a bar chart of most frequent words used as code comments.

## Results

The results given for our experiments can be found in the [results](results/) folder.
