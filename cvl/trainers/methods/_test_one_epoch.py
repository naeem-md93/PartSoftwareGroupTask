import gc
import torch
import numpy as np
from tqdm import tqdm
from rich.live import Live

def test_one_classification_epoch(
    epoch: int,
    mode: str,
    device: torch.device,
    dataloader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    evaluator,
    logger
):
    H = {"L": [], "A": []}
    cm_title = f"M: {mode} | E: {epoch} |" + " B: {0}/" + str(len(dataloader)) + " | L: {1} | A: {2} |"

    with Live(evaluator.build_cm_table(title=cm_title.format(0, 0, 0)), refresh_per_second=1) as live:
        for bn, data in enumerate(dataloader):

            images = data["image"].to(device)
            labels = data["label"].to(device)

            with torch.no_grad():
                logits = model(images)

            loss = criterion(logits, labels)

            gc.collect()
            torch.cuda.empty_cache()

            evaluator.update(labels, logits)

            acc = evaluator.get_accuracy()
            H["L"].append(loss.item())
            H["A"].append(acc)

            logger.add_scalar(f"{mode}/batch/loss", loss.item())
            logger.add_scalar(f"{mode}/batch/accuracy", acc)

            live.update(evaluator.build_cm_table(title=cm_title.format(
                bn,
                round(np.mean(H['L']), 2),
                round(np.mean(H['A']), 2),
            )))

        H["CM"] = evaluator.get_confusion_matrix()

        logger.add_scalar(f"{mode}/epoch/loss", np.mean(H["L"]))
        logger.add_scalar(f"{mode}/epoch/accuracy", np.mean(H["A"]))

        return H