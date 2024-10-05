import os
import logging

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torch import distributed as dist
from torch.multiprocessing import Process


class CliendDataset(Dataset):
    def __init__(self, org_ds, sample_idx):
        self.org_ds = org_ds
        self.sample_idx = sample_idx

    def __getitem__(self, idx):
        return self.org_ds[self.sample_idx[idx]]

    def __len__(self):
        return len(self.sample_idx)


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, 2)
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# define the worker behavior
def worker(rank, world_size):
    # sys.stdout = open(f"stdout-{rank}-{world_size}.log", "w")
    print(f"Rank {rank}-{world_size} is running")

    # set up the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    ds_path = "datasets"
    org_ds = datasets.MNIST(ds_path, train=True, transform=transforms.ToTensor())

    global_idx = torch.randperm(len(org_ds)).reshape(world_size, -1)
    global_idx = [global_idx[i] for i in range(world_size)] if rank == 0 else None
    local_idx = torch.empty((len(org_ds) // world_size,), dtype=torch.int64)
    dist.scatter(local_idx, global_idx, src=0)

    cliend_ds = CliendDataset(org_ds, local_idx)
    cliend_dl = DataLoader(cliend_ds, batch_size=32, shuffle=True)

    model = LeNet5()
    for epoch in range(10):
        print(f"Rank {rank}-{world_size} epoch {epoch}")
        for i, (data, target) in enumerate(cliend_dl):
            output = model(data)
            loss = nn.functional.cross_entropy(output, target)
            loss.backward()
        if epoch % 2 == 0:
            print(f"Rank {rank}-{world_size} epoch {epoch} aggregate")
            with torch.no_grad():
                for param in model.parameters():
                    dist.all_reduce(param, op=dist.ReduceOp.SUM)
                    param /= world_size

    dist.destroy_process_group()


def main():
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    os.system("lsof -ti :12355 | xargs kill -9")
    word_size = 10
    processes = []
    for rank in range(word_size):
        p = Process(target=worker, args=(rank, word_size))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    logger = logging.getLogger(__name__)
    logger.info("Start")
    main()
    logger.info("End")
