from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
import json

# Functions to compute training confusion matrices and graphs for models.

# Format data
def format_data(fold_results):
    # Combine all folds
    all_predictions = []
    all_targets = []

    train_loss = []
    val_loss = []

    for result in fold_results:
        all_predictions.extend(result["final_predictions"])
        all_targets.extend(result["final_targets"])
        train_loss.append(result["training_loss"])
        val_loss.append(result["validation_loss"])

    return all_predictions, all_targets, train_loss, val_loss


# Generate graphs
def plot_loss(train_loss, val_loss, model_name, hp_set, results_dir):
    train_loss_matrix = np.array(train_loss)
    if val_loss is not None:
        val_loss_matrix = np.array(val_loss)

    # Get stats
    epochs = range(1, train_loss_matrix.shape[1] + 1)
    train_mean = train_loss_matrix.mean(axis=0)
    train_std = train_loss_matrix.std(axis=0)
    if val_loss is not None:
        val_mean = val_loss_matrix.mean(axis=0)
        val_std = val_loss_matrix.std(axis=0)

    plt.figure(figsize=(10, 6))

    num_folds = train_loss_matrix.shape[0]
    for i in range(num_folds):
        plt.plot(epochs, train_loss_matrix[i], color="blue", alpha=0.15, linewidth=1)
        if val_loss is not None:
            plt.plot(epochs, val_loss_matrix[i], color="orange", alpha=0.15, linewidth=1)

    # Plot main curves
    plt.plot(epochs, train_mean, label="Mean Train Loss", color="blue", linewidth=2)
    plt.fill_between(epochs, train_mean - train_std, train_mean + train_std, alpha=0.2, color="blue")
    if val_loss is not None:
        plt.plot(epochs, val_mean, label="Mean Validation Loss", color="orange", linewidth=2)
        plt.fill_between(epochs, val_mean - val_std, val_mean + val_std, alpha=0.2, color="orange")
    
    # Finalize graph labels and ensure formatting
    plt.title(f"5-Fold Loss Curves Average: {model_name} ({hp_set})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"{model_name}_{hp_set}_loss_curve.png"))
    plt.close()

# Generate test plot
def plot_test(train_loss, test_loss, model_name, hp_set, results_dir):
    test_loss = test_loss[0]
    num_epochs = range(1, len(train_loss) + 1)
    final_epoch = len(train_loss)

    # Plot curves and final test point
    plt.figure(figsize=(10, 6))
    plt.plot(num_epochs, train_loss, label="Mean Train Loss", color="blue", linewidth=2)
    plt.scatter(final_epoch, test_loss, color="red", s=100, zorder=5, edgecolor="black", linewidth=1, label="Test Loss")
    plt.text(final_epoch + 0.5, test_loss, f'Test Loss: {test_loss:.4f}', color="red", va='center')

    # Finalize graph labels and ensure formatting
    plt.title(f"Final Training Loss Curve: {model_name} ({hp_set})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"{model_name}_{hp_set}_final_loss_curve.png"))
    plt.close()

# Generate confusion matrix
# helpful understanding: https://www.geeksforgeeks.org/machine-learning/confusion-matrix-machine-learning
def confusion_matrix_gen(cm_predictions, cm_targets, model_name, hp_set, city_names, results_dir):
    cm = confusion_matrix(cm_targets, cm_predictions)
    plt.figure(figsize=(12, 10))
    # Not sure about colour yet
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=city_names, yticklabels=city_names)
    # sns.heatmap(cm, annot=True, fmt="d", cmap=None, cbar=False, linewidths=0.5, linecolor="black" xticklabels=city_names, yticklabels=city_names)
    plt.title(f"Confusion Matrix: {model_name} ({hp_set})")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"{model_name}_{hp_set}_confusion_matrix.png"))
    plt.close()

# Generate accuracy and f1 reports
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html
def accuracy_f1_metrics(all_predictions, all_targets, class_names, model_name, hp_set, results_dir):
    # Use sklearn to get all the f1 needs and create report
    precision, recall, f1, support = precision_recall_fscore_support(all_targets, all_predictions, average=None)
    accuracy = np.sum(np.array(all_targets) == np.array(all_predictions)) / len(all_targets)

    metrics_df = pd.DataFrame({
        "Class": class_names,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "Support": support
    })

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(all_targets, all_predictions, average="macro")

    summary_row = pd.DataFrame({
        "Class": ["MACRO AVG", "ACCURACY"],
        "Precision": [macro_p, accuracy],
        "Recall": [macro_r, accuracy],
        "F1-Score": [macro_f1, accuracy],
        "Support": [len(all_targets), len(all_targets)]
    })

    final_df = pd.concat([metrics_df, summary_row], ignore_index=True)
    final_df.to_csv(os.path.join(results_dir, f"{model_name}_{hp_set}_metrics.csv"), index=False)

if __name__ == "__main__":
    class_names = [
        "Calgary", "Charlottetown", "Edmonton", "Halifax", "Hamilton",
        "Kitchener-Waterloo", "Montreal", "Ottawa-Gatineau", "Quebec City", "Saskatoon",
        "St Johns", "Toronto", "Vancouver", "Victoria", "Winnipeg",
    ]

    results_path = "vit_results/swinv2_base_window12_192_final_training_results_0.json"
    model_name = "swinv2_base_window12_192"
    hp_set = "1"
    results_dir = "final_training_results"

    os.makedirs(results_dir, exist_ok=True)

    with open(results_path, "r") as f:
        results_dict = json.load(f)

    # Format data (handles single dict)
    all_predictions, all_targets, train_loss, val_loss = format_data(results_dict)

    print(f"Total predictions: {len(all_predictions)}, Total targets: {len(all_targets)}")
    if train_loss and val_loss:
        print(f"Number of epochs: {len(train_loss[0])}")

    # Plot loss curves
    if train_loss and val_loss and all(len(l) > 0 for l in train_loss + val_loss):
        plot_loss(train_loss, val_loss, model_name, hp_set, results_dir)
    else:
        if train_loss and all(len(l) > 0 for l in train_loss):
            print("No validation loss data found; skipping validation loss plot.")
            plot_loss(train_loss, None, model_name, hp_set, results_dir)
        else:
            print("No loss data found; skipping loss plot.")

    # Confusion matrix
    confusion_matrix_gen(
        cm_predictions=all_predictions,
        cm_targets=all_targets,
        model_name=model_name,
        hp_set=hp_set,
        city_names=class_names,
        results_dir=results_dir
    )

    # Metrics CSV
    accuracy_f1_metrics(
        all_predictions,
        all_targets,
        class_names,
        model_name,
        hp_set,
        results_dir
    )

    print("Analysis complete. Files saved in:", results_dir)
