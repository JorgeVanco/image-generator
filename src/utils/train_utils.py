from torch.nn.utils.clip_grad import clip_grad_norm_


def get_gradient_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm**0.5
    return total_norm


def train_loop(
    dataloader, model, train_step, optimizer, scheduler, epochs=5, verbose=True
) -> tuple[list, list]:
    model.train()
    size = len(dataloader.dataset)
    losses = []
    gradients = []
    try:
        for epoch in range(epochs):

            if verbose:
                print(f"Epoch {epoch+1}\n-------------------------------")

            for batch, (X, label) in enumerate(dataloader):
                # Compute prediction error
                loss = train_step(model, X)

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()

                clip_grad_norm_(model.parameters(), 1)
                optimizer.step()

                gradients.append(get_gradient_norm(model))
                losses.append(loss.item())

                if verbose and batch % 32 == 0:
                    loss, current = loss.item(), batch * len(X)
                    print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

            scheduler.step()

    except KeyboardInterrupt:
        print("Training interrupted.")

    return losses, gradients
