from transformers import AutoModelForTokenClassification, AutoTokenizer, TrainingArguments,\
    Trainer, DataCollatorForTokenClassification
from datasets import load_from_disk
import numpy as np
import torch
import argparse
import os


# noinspection PyShadowingNames
def parse_arguments():
    # a helper function to parse arguments
    parser = argparse.ArgumentParser(description='Get the directory to save the model.')

    # Add the --save-dir argument
    parser.add_argument('--save-dir', type=str, default=None, help='Specify the directory for saving.')

    args = parser.parse_args()

    return args


def align_labels_with_tokens(labels, word_ids):
    """
    convert labeled words to labeled tokens
    :param labels: word labels
    :param word_ids: list of word ids for each token
    :return: token labels
    """
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # a new word
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # special token, typically set to -100
            new_labels.append(-100)
        else:
            # same word as previous token
            label = labels[word_id]
            new_labels.append(label)

    return new_labels


def tokenize_and_align_labels(examples):
    """
    tokenize the samples and align the labels
    :param examples: dataset samples
    :return: tokenized samples
    """
    # apply tokenizer to the tokens
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
    )
    all_labels = examples["fine_ner_tags"]
    new_labels = []
    # for each sentence in the dataset align the labels for tokens
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs


def compute_metrics1(eval_preds):
    """
    compute the metrics for a batch of data;
    used during the model evaluation
    :param eval_preds: a batch of data predictions
    :return: precision, recall, and f1-score for class 1 (the mountains)
    """
    logits, labels = eval_preds
    # get the most probable class
    predictions = np.argmax(logits, axis=-1)

    # ignore -100 - padding token
    true_labels = [[elem for elem in label if elem != -100] for label in labels]
    true_predictions = [
        [p for (p, elem) in zip(prediction, label) if elem != -100]
        for prediction, label in zip(predictions, labels)
    ]

    y = true_labels
    y_hat = true_predictions

    # compute the metrics
    tp = 0
    fp = 0
    fn = 0
    for i in range(len(y)):
        for j in range(len(y[i])):
            if y[i][j] == y_hat[i][j]:
                if y[i][j] == 1:
                    tp += 1
            else:
                if y[i][j] == 1:
                    fn += 1
                else:
                    fp += 1

    # avoid division by 0
    pr = tp / (tp + fp) if tp + fp > 0 else 0.
    re = tp / (tp + fn) if tp + fn > 0 else 0.
    f1 = 2*pr*re / (pr + re) if pr + re > 0 else 0.

    return {
        "precision_class1": pr,
        "recall_class1": re,
        "f1_class1": f1,
    }


# noinspection PyShadowingNames
class CustomTrainer(Trainer):
    """
    a Trainer with a custom loss function, to account for different class weights
    """
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        compute the loss with respect to a batch of inputs
        :param model: the model
        :param inputs: the inputs to be fed into the model
        :param return_outputs: True, if the outputs should be returned, False, if only the loss
        :return: the loss and the model outputs, if required
        """
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get('logits')

        # ignore -100 label
        labels[labels == -100] = 0

        # the ratio of 0-tokens to 1-tokens (approximate value)
        ratio = 12.

        # Move the weight tensor to the same device as the model's parameters
        weight_tensor = torch.tensor([1., ratio], device=logits.device)

        # hinge loss
        loss_fct = torch.nn.MultiMarginLoss(weight=weight_tensor)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


# noinspection PyShadowingNames
def train_model(model, tokenizer, train_set, val_set, data_collator, lr=2e-5, epochs=1.):
    """
    train the model with specified configurations
    :param model: the model
    :param tokenizer: the tokenizer
    :param train_set: the data for training
    :param val_set: the data for validation
    :param data_collator: the data_collator
    :param lr: the learning rate
    :param epochs: the number of epochs
    """
    os.makedirs('./models/logs', exist_ok=True)

    args = TrainingArguments(
        "./models/logs",
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=lr,
        num_train_epochs=epochs,
        weight_decay=0.01,
        report_to=None,
    )

    trainer = CustomTrainer(
        model=model,
        args=args,
        train_dataset=train_set,
        eval_dataset=val_set,
        data_collator=data_collator,
        compute_metrics=compute_metrics1,
        tokenizer=tokenizer,
    )

    trainer.train()


if __name__ == '__main__':
    # parse cmd arguments
    args = parse_arguments()
    save_dir = args.save_dir

    # Load the tokenizer and the model from huggingface
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    model = AutoModelForTokenClassification.from_pretrained("bert-base-cased")

    # load the dataset splits
    train_loaded = load_from_disk('./data/train_data')
    val_loaded = load_from_disk('./data/val_data')

    # apply tokenization
    tokenized_train = train_loaded.map(
        tokenize_and_align_labels,
        batched=True
    )
    tokenized_val = val_loaded.map(
        tokenize_and_align_labels,
        batched=True
    )

    # create a data collator, which does the following:
    # 1. pad the label sequences
    # 2. mask out the padded labels
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    # the exact num_epochs which was used to train the final model
    train_model(model, tokenizer, tokenized_train, tokenized_val, data_collator, epochs=4)

    if save_dir is not None:
        # create the directory if not exists
        os.makedirs(save_dir, exist_ok=True)

        model.save_pretrained(os.path.join(save_dir, 'model'))
        tokenizer.save_pretrained(os.path.join(save_dir, 'tokenizer'))
