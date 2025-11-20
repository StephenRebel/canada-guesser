from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns

# Functions to compute training confusion matrices and graphs for models.

# Format data
def format_data(fold_results)
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
    val_loss_matrix = np.array(val_loss)

    # Get stats
    epochs = range(1, train_loss_matrix.shape[1] + 1)
    train_mean = train_loss_matrix.mean(axis=0)
    train_std = train_loss_matrix.std(axis=0)
    val_mean = val_loss_matrix.mean(axis=0)
    val_std = val_loss_matrix.std(axis=0)

    plt.figure(figsize=(10, 6))

    num_folds = train_loss_matrix.shape[0]
    for i in range(num_folds):
        plt.plot(epochs, train_loss_matrix[i], color="blue", alpha=0.15, linewidth=1)
        plt.plot(epochs, val_loss_matrix[i], color="orange", alpha=0.15, linewidth=1)

    # Plot main curves
    plt.plot(epochs, train_mean, label="Mean Train Loss", color="blue", linewidth=2)
    plt.fill_between(epochs, train_mean - train_std, train_mean + train_std, alpha=0.2, color="blue")
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


# Generate confusion matrix
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
def accuracy_f1_metrics(all_predictions, all_targets, model_name, hp_set, results_dir):
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
