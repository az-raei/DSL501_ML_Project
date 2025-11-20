!pip install transformers accelerate datasets evaluate

import torch
from transformers import (
    XLMRobertaTokenizer,
    XLMRobertaForSequenceClassification,
    TrainingArguments,
    Trainer
)
import numpy as np
import evaluate

labels = ["Emotional Acknowledgement",
          "Validation",
          "Reflective Understanding",
          "Supportive Reframing",
          "Action-oriented Guidance"]

label_to_id = {l:i for i,l in enumerate(labels)}
id_to_label = {i:l for l,i in label_to_id.items()}


def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=256
    )


def compute_metrics(eval_pred):
    preds, labels_true = eval_pred
    preds = np.argmax(preds, axis=1)

    f1 = evaluate.load("f1").compute(
        predictions=preds, references=labels_true, average="macro"
    )
    precision = evaluate.load("precision").compute(
        predictions=preds, references=labels_true, average="macro"
    )
    recall = evaluate.load("recall").compute(
        predictions=preds, references=labels_true, average="macro"
    )

    return {
        "f1": f1["f1"],
        "precision": precision["precision"],
        "recall": recall["recall"],
    }


def train_asem(train_df, val_df):

    global tokenizer
    tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")

    train_ds = {
        "text": (train_df["post"] + " [SEP] " + train_df["response"]).tolist(),
        "labels": train_df["label"].apply(lambda x: label_to_id[x]).tolist(),
    }

    val_ds = {
        "text": (val_df["post"] + " [SEP] " + val_df["response"]).tolist(),
        "labels": val_df["label"].apply(lambda x: label_to_id[x]).tolist(),
    }

    from datasets import Dataset
    train_ds = Dataset.from_dict(train_ds).map(tokenize)
    val_ds = Dataset.from_dict(val_ds).map(tokenize)

    # handle imbalance
    class_counts = train_df["label"].value_counts().to_dict()
    weights = torch.tensor([1/class_counts[id_to_label[i]] for i in range(len(labels))])
    loss_fct = torch.nn.CrossEntropyLoss(weight=weights.to("cuda"))

    model = XLMRobertaForSequenceClassification.from_pretrained(
        "xlm-roberta-base",
        num_labels=len(labels)
    )

    def custom_loss(model, inputs, return_outputs=False):
        labels = inputs["labels"]
        outputs = model(**inputs)
        logits = outputs.logits
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

    training_args = TrainingArguments(
        output_dir="./asem_xlmr",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        num_train_epochs=4,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        weight_decay=0.01,
        logging_steps=20,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        loss_func=custom_loss
    )

    trainer.train()
    trainer.save_model("./asem_xlmr_best")
