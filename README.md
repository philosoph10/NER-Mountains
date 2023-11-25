# Mountain Recognition NER Project

![Mount Everest](https://cdn.britannica.com/17/83817-050-67C814CD/Mount-Everest.jpg)

## Overview

This project addresses the task of mountain recognition through Named Entity Recognition (NER). The primary objective is to identify mountainous regions within a given text. The final model has been trained on the Few-NERD dataset and achieves a notable F1-score of 0.87 on class 1 (mountains) in the test set.

## Dataset

The dataset is available in the `data` folder, which includes three subfolders: `train_data`, `test_data`, and `val_data`. You can load the datasets using the `datasets.load_from_disk` method as demonstrated below:

```python
import datasets

# Load test data
test_loaded = datasets.load_from_disk('./data/test_data')
```

## Dataset Creation

The dataset creation process involves two main steps:

1. **Discarding Tokens:**
   - Discard all tokens except for mountain tokens.
   - Label mountains as 1, and everything else as 0.

2. **Reducing Dataset:**
   - Keep all samples with mountains.
   - Include a small fraction of other samples.

## Model

The chosen model for this task is `bert-base-cased`. The model has been trained with the following configurations:

- Number of epochs: 4
- Hinge loss with weights assigned to classes

## Hugging Face

The trained model, along with its corresponding tokenizer, is available on Hugging Face. You can access it through the following link:

[huggingsaurusRex/bert-base-uncased-for-mountain-ner](https://huggingface.co/huggingsaurusRex/bert-base-uncased-for-mountain-ner)

## Files

The repository contains the following files:

- **dataset_creation.ipynb:** Walks you through the dataset creation process.
- **demo.ipynb:** Demonstrates the evaluation on the test set and inference on hand-made samples.
- **training.py:** Python script for training the model.
- **inference.py:** Python script for performing inference. Accepts the following arguments:
  - `--input-path:` Path to a TXT file for model input.
  - `--output-path:` Directory to store the output file.

The output will be a TXT file with all the mountain names enclosed into \<mountain\> tag, in XML style.

Feel free to explore and utilize these resources to enhance your understanding of the project!
