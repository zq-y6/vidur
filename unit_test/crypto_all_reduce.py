import torch
import torch.distributed as dist
import os
from multiprocessing import Process


def gather_all_reduce(send, recv):
    dist.all_gather_into_tensor(recv, send)
    send_size = send.numel()
    send = recv.view(dist.get_world_size(), send_size).sum(dim=0)
    return send


def ring_all_reduce(send, recv):
    # 2(n - 1) serialized encryption + decryption for each GPU
    rank = dist.get_rank()
    size = dist.get_world_size()
    send_buff = send.clone()
    recv_buff = send.clone()
    accum = send.clone()

    left = ((rank - 1) + size) % size
    right = (rank + 1) % size

    # init: 1, 2, 3, 4 ----> 10, 10, 10, 10
    # s1r:  4, 1, 2, 3
    # s1s:  1, 2, 3, 4
    # s1a:  5, 3, 5, 7
    # s2r:  4, 1, 2, 3
    # s2s:  3, 4, 1, 2
    # s2a:  8, 7, 6, 9
    # s3r:  2, 3, 4, 1
    # s3a:  10, 10, 10, 10

    for i in range(size - 1):
        if i % 2 == 0:
            # Send send_buff
            if rank % 2 == 0:
                send_req = dist.isend(send_buff, right)
                dist.recv(recv_buff, left)
                send_req.wait()
            else:
                recv_req = dist.irecv(recv_buff, left)
                dist.send(send_buff, right)
                recv_req.wait()
            accum[:] += recv_buff[:]
        else:
            # Send recv_buff
            if rank % 2 == 0:
                send_req = dist.isend(recv_buff, right)
                dist.recv(send_buff, left)
                send_req.wait()
            else:
                recv_req = dist.irecv(send_buff, left)
                dist.send(recv_buff, right)
                recv_req.wait()
            accum[:] += send_buff[:]
    recv[:] = accum[:]
    return recv


def test(fn, dup):
    world_size = int(os.environ["WORLD_SIZE"])
    tensor = torch.tensor([rank + 1.0], device="cuda")
    recv = torch.zeros_like(tensor) if not dup else tensor.repeat(world_size)
    print(f"Process {rank} before all-reduce: {tensor}")
    tensor = fn(tensor, recv)
    print(f"Process {rank} after all-reduce: {tensor}")


if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"

    assert "RANK" in os.environ
    assert "WORLD_SIZE" in os.environ

    dist.init_process_group("nccl")
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(rank)

    test(ring_all_reduce, False)
    test(gather_all_reduce, True)

    dist.destroy_process_group()
