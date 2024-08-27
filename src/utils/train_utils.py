from torch.nn.utils.clip_grad import clip_grad_norm_
import os
from utils import download_dataset
from data_processing.Dataset import PixelDataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def get_gradient_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm**0.5
    return total_norm


def get_dataset_dataloader(
    root_path: str, data_dir: str, batch_size: int, num_workers: int, logger=None
) -> DataLoader:
    dataset_path = os.path.join(root_path, data_dir)
    sprites_path = os.path.join(dataset_path, "sprites.npy")
    labels_path = os.path.join(dataset_path, "sprites_labels.npy")

    if not os.path.exists(sprites_path):
        if logger:
            logger.warning("Dataset not found. Downloading...")
        download_dataset(dataset_path)

    dataset = PixelDataset(sprites_path=sprites_path, labels_path=labels_path)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    return dataset, dataloader


def train_loop(
    dataloader,
    model,
    train_step,
    optimizer,
    scheduler,
    epochs=5,
    logger=None,
    writer=None,
    device="cpu",
) -> tuple[list, list]:
    model.train()
    model.to(device)
    size = len(dataloader.dataset)
    losses = []
    gradients = []
    try:
        for epoch in tqdm(range(epochs)):
            running_loss = 0.0
            for batch_idx, (X, label) in enumerate(dataloader):
                # Compute prediction error
                X = X.to(device)
                loss = train_step(model, X)

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()

                clip_grad_norm_(model.parameters(), 1)
                optimizer.step()

                gradients.append(get_gradient_norm(model))
                losses.append(loss.item())
                running_loss += loss.item()
                if batch_idx % 32 == 0:
                    loss, current = loss.item(), batch_idx * len(X)
                    if logger:
                        logger.info(
                            f"Epoch [{epoch:>5d}/{epochs}], Batch [{batch_idx:>5d}/{len(dataloader)}], Samples [{current:>5d}/{size}], Loss: {loss:.4f}"
                        )
                    if writer:
                        writer.add_scalar(
                            "Training loss", loss, epoch * len(dataloader) + batch_idx
                        )

            avg_loss = running_loss / len(dataloader)
            if logger:
                logger.info(
                    f"Epoch [{epoch:>5d}/{epochs}], Average Loss: {avg_loss:.4f}"
                )
            if writer:
                writer.add_scalar("Average training loss", avg_loss, epoch)
            scheduler.step()

    except KeyboardInterrupt:
        logger.warning("Training interrupted.")

    return losses, gradients
