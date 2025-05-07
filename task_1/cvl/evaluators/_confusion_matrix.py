from typing import List
import torch
import numpy as np
from rich.table import Table


class ConfusionMatrixEvaluator:
    def __init__(self, mode: str, class_labels: List[str], name: str, num_classes: int, threshold=0.5) -> None:
        self.mode = mode
        self.class_labels = class_labels
        self.num_classes = num_classes
        self.threshold = threshold
        self.confusion_matrix = np.zeros(shape=(self.num_classes, self.num_classes), dtype=np.int64)

    def reset(self) -> None:
        self.confusion_matrix = np.zeros(shape=(self.num_classes, self.num_classes), dtype=np.int64)

    def get_confusion_matrix(self) -> np.ndarray:
        return self.confusion_matrix

    def update(self, labels: torch.Tensor, logits: torch.Tensor) -> None:
        logits = logits.detach().argmax(dim=1).cpu().numpy()
        labels = labels.detach().argmax(dim=1).cpu().numpy()

        for i in range(len(logits)):
            self.confusion_matrix[logits[i], labels[i]] += 1

    def get_accuracy(self) -> float:
        numerator = self.confusion_matrix[range(self.num_classes), range(self.num_classes)].sum()
        denominator = self.confusion_matrix.sum() + 1e-9
        return numerator / denominator

    def build_cm_table(self, title: str) -> Table:
        """Builds a Rich table displaying the confusion matrix with row/column labels."""
        table = Table(title=title)
        # First column header (an empty cell or a label for rows)
        table.add_column("Pred \\ Actual", justify="center", style="bold")
        # Add a column for each predicted label
        for label in self.class_labels:
            table.add_column(label, justify="center")
        # Add rows: each row starts with the true label name and then values from the matrix
        for i, row_label in enumerate(self.class_labels):
            row = [row_label]  # first column: the actual label for this row
            for j in range(len(self.class_labels)):
                row.append(str(self.confusion_matrix[i, j]))
            table.add_row(*row)
        return table

