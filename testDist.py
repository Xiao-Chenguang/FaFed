import os

import torch
from torch import distributed as dist
from torch.multiprocessing import Process


# define the worker behavior
def worker(rank, world_size):
    # sys.stdout = open(f"stdout-{rank}-{world_size}.log", "w")
    print(f"Rank {rank}-{world_size} is running")

    # set up the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    idxs = torch.randint(0, 100, (dist.get_world_size(),))
    idx = torch.tensor([rank])
    dist.scatter(idx, idxs, src=0)

    print(f"Rank {rank}-{world_size} generate idxs {idxs} and get {idx}")

    # all processes run the same code
    tensor = torch.tensor([rank])
    print(f"Rank {rank}-{world_size} has data {tensor[0]}")
    # dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    # print(f"Rank {rank}-{world_size} reduce data to {tensor[0]}")
    dist.broadcast(tensor, src=0)
    print(f"Rank {rank}-{world_size} reduce data to {tensor[0]}")
    print(f"Rank {rank}-{world_size} finished")

    dist.destroy_process_group()


def main():
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    word_size = 4
    processes = []
    for rank in range(word_size):
        p = Process(target=worker, args=(rank, word_size))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == "__main__":
    main()
