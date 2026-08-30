import copy
import time

import torch

from torch import nn
from torch.utils.data import DataLoader


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    loss_function: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    number_of_epochs: int = 100,
    patience: int = 15,
) -> dict:
    training_losses = []
    validation_losses = []

    best_validation_loss = float("inf")
    best_model_state = None
    epochs_without_improvement = 0

    training_start_time = time.perf_counter()

    for epoch in range(number_of_epochs):
        model.train()

        total_training_loss = 0.0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            predictions = model(batch_X)

            loss = loss_function(
                predictions,
                batch_y,
            )

            loss.backward()
            optimizer.step()

            total_training_loss += (
                loss.item()
                * batch_X.size(0)
            )

        average_training_loss = (
            total_training_loss
            / len(train_loader.dataset)
        )

        training_losses.append(
            average_training_loss
        )

        model.eval()

        total_validation_loss = 0.0

        with torch.no_grad():
            for batch_X, batch_y in validation_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)

                predictions = model(batch_X)

                validation_loss = loss_function(
                    predictions,
                    batch_y,
                )

                total_validation_loss += (
                    validation_loss.item()
                    * batch_X.size(0)
                )

        average_validation_loss = (
            total_validation_loss
            / len(validation_loader.dataset)
        )

        validation_losses.append(
            average_validation_loss
        )

        if (
            average_validation_loss
            < best_validation_loss
        ):
            best_validation_loss = (
                average_validation_loss
            )

            best_model_state = copy.deepcopy(
                model.state_dict()
            )

            epochs_without_improvement = 0

        else:
            epochs_without_improvement += 1

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:3d} | "
                f"Train: {average_training_loss:.6f} | "
                f"Validation: {average_validation_loss:.6f}"
            )

        if (
            epochs_without_improvement
            >= patience
        ):
            print(
                f"Early stopping at epoch "
                f"{epoch + 1}"
            )
            break

    training_time = (
        time.perf_counter()
        - training_start_time
    )

    if best_model_state is None:
        raise RuntimeError(
            "No best model state was saved."
        )

    model.load_state_dict(
        best_model_state
    )

    return {
        "model": model,
        "training_losses": training_losses,
        "validation_losses": validation_losses,
        "best_validation_loss":
            best_validation_loss,
        "training_time_seconds":
            training_time,
        "epochs_trained":
            len(training_losses),
    }
