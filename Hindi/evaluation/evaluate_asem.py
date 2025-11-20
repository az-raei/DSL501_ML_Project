import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(model, tokenizer, test_df):

    texts = (test_df["post"] + " [SEP] " + test_df["response"]).tolist()
    labels_true = test_df["label"].apply(lambda x: label_to_id[x]).tolist()

    preds = []
    for t in texts:
        inp = tokenizer(t, return_tensors="pt", truncation=True).to(model.device)
        out = model(**inp).logits
        preds.append(int(out.argmax(dim=1)))

    print("\nclassification report:")
    print(classification_report(labels_true, preds, target_names=labels))

    cm = confusion_matrix(labels_true, preds)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, xticklabels=labels, yticklabels=labels, fmt="d")
    plt.title("Confusion Matrix")
    plt.show()
