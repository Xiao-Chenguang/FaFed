import logging
from typing import Sized

import torch
import torch.multiprocessing as mp
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor


class CliendDataset(Dataset):
    def __init__(self, org_ds: Dataset, sample_idx: torch.Tensor):
        self.org_ds = org_ds
        self.sample_idx = sample_idx

    def __getitem__(self, idx):
        return self.org_ds[self.sample_idx[idx]]

    def __len__(self):
        return len(self.sample_idx)


def gen_fl_dls(org_ds: Dataset, world_size: int) -> list[DataLoader]:
    """Generate federated data loaders for each worker.

    Args:
        org_ds (Dataset): The original dataset.
        world_size (int): The number of workers.

    Returns:
        list[DataLoader]: A list of federated data loaders.
    """
    assert isinstance(org_ds, Sized), "org_ds must implement the __len__ method"
    """Generate federated data loaders for each worker."""
    total_samples = len(org_ds) - (len(org_ds) % world_size)
    global_idx = torch.randperm(total_samples).reshape(world_size, -1)
    return [
        DataLoader(CliendDataset(org_ds, global_idx[i]), batch_size=32, shuffle=True)
        for i in range(world_size)
    ]


def train_client(cid, model, data_loader, logger):
    logger.debug(f"Starting task for client {cid}")

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.01)
    cost = 0.0
    for epoch in range(3):
        for x, y in data_loader:
            logits = model(x)
            loss = loss_fn(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            cost += loss.item()
    cost /= len(data_loader) * 3

    # Simulate local training (replace with actual training)
    logger.debug(f"loss: {cost}")
    return cid, model, cost


# Function that simulates a client process
def client(worker_id, task_queue, result_queue):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(processName)s] %(message)s",
    )
    logger = logging.getLogger(f"Worker-{worker_id}")
    logger.debug(f"Worker {worker_id} starting...")

    while True:
        task = task_queue.get()  # Wait for a task from the server

        if task == "STOP":
            logger.debug(f"Worker {worker_id} stopping.")
            break  # Exit if the server sends a stop signal

        # Get the latest global model from the task queue
        cid, model, data_loader = task

        cid, model, cost = train_client(cid, model, data_loader, logger)

        # Send the local model update back to the server
        result_queue.put((cid, model, cost))


# Server function
def server(
    global_model,
    data_loaders,
    test_loader,
    num_clients,
    num_process,
    num_global_rounds,
    task_queue,
    result_queue,
):
    logger = logging.getLogger("Server")
    logger.info("Server starting...")

    for global_round in range(num_global_rounds):
        logger.info(f"Starting global round {global_round+1}/{num_global_rounds}")

        # Send the current global model to all clients via task queue
        if num_process > 0:
            for cid in range(num_clients):
                task_queue.put((cid, global_model, data_loaders[cid]))

        # Collect updates from all clients
        updates = []
        costs = []
        for cid in range(num_clients):
            if num_process > 0:
                client_id, local_update, cost = result_queue.get()
            else:
                client_id, local_update, cost = train_client(
                    cid, global_model, data_loaders[cid], logger
                )
            logger.info(f"Received Client {client_id} cost {cost:.4f}")
            updates.append(local_update)
            costs.append(cost)

        # Aggregate updates and update the global model
        global_model[1].weight.data = torch.mean(
            torch.stack([update[1].weight.data for update in updates]), dim=0
        )
        global_model[1].bias.data = torch.mean(
            torch.stack([update[1].bias.data for update in updates]), dim=0
        )

        # Evaluate the global model
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = global_model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total
        logger.info(f"Global model accuracy: {accuracy}")

    # Send stop signals to clients after training completes
    for _ in range(num_clients):
        task_queue.put("STOP")

    logger.info("Training completed.")


if __name__ == "__main__":
    # Configure logging for the main process and child processes
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(processName)s] %(message)s",
    )

    # Set the number of global rounds
    num_global_rounds = 5
    num_clients = 8
    num_process = 4

    # Initialize the global model
    global_model = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
    dataset = MNIST("datasets", train=True, transform=ToTensor())
    testset = MNIST("datasets", train=False, transform=ToTensor())
    data_loaders = gen_fl_dls(dataset, num_clients)
    test_loader = DataLoader(testset, batch_size=128)

    # Create queues for task distribution and result collection
    task_queue = mp.Queue()
    result_queue = mp.Queue()

    # Start client processes
    processes = []
    for worker_id in range(num_process):
        args = (worker_id, task_queue, result_queue)
        p = mp.Process(target=client, args=args)
        p.start()
        processes.append(p)

    # Start the server
    server(
        global_model,
        data_loaders,
        test_loader,
        num_clients,
        num_process,
        num_global_rounds,
        task_queue,
        result_queue,
    )

    # Wait for all client processes to finish
    for p in processes:
        p.join()
