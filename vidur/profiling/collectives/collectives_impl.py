from typing import Callable

import torch
import extension_cpp

WARMUP_STEPS = 5
GRAPH_STEPS = 3


class GraphedCollective:
    def __init__(
        self,
        num_workers: int,
        size: int,
        collective: str = "all_reduce",
        disable_graph: bool = False,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        self._size = size
        self._disable_graph = disable_graph
        self._collective_fn = self._get_collective_fn(collective)

        self._buffer = torch.empty(
            size=(size,),
            dtype=dtype,
            device="cuda",
        )
        self._gather_buffer = None
        if collective == "all_gather":
            self._gather_tensor = torch.empty(
                size=(size * num_workers,),
                dtype=dtype,
                device="cuda",
            )
        elif collective == "reduce_scatter":
            self._reduce_buffer = torch.empty(
                size=(size * num_workers,),
                dtype=dtype,
                device="cuda",
            )
        elif collective == "my_ring_all_reduce":
            self._iv = torch.empty(size=(3,), dtype=torch.uint32, device="cuda")
            buffer_size_in_uint32 = size * self._buffer.element_size() / self._iv.element_size()
            self._rek = torch.empty(size=(56,), dtype=torch.uint32, device="cuda")
            self._ct = torch.empty(size=(int(2 * buffer_size_in_uint32),), dtype=torch.uint32, device="cuda")   # top half for ciphertext, bottom for sig
            self._pt = torch.empty(size=(int(2 * buffer_size_in_uint32),), dtype=torch.uint32, device="cuda")   # top half for plaintext, bottom for auth
            self._nw = num_workers
            self._h = torch.randint(
                low=0,
                high=2**32,
                size=(2,),
                dtype=torch.uint64,
                device="cuda"
            )

        if not self._disable_graph:
            self._graph = self._build_graph()

    def _run_all_reduce(self):
        torch.distributed.all_reduce(self._buffer)

    def _run_all_gather(self):
        torch.distributed.all_gather_into_tensor(self._gather_tensor, self._buffer)

    def _run_broadcast(self):
        torch.distributed.broadcast(self._buffer, 0)

    def _run_send_recv(self):
        if torch.distributed.get_rank() == 0:
            torch.distributed.send(self._buffer, 1)
        else:
            torch.distributed.recv(self._buffer, 0)

    def _run_reduce_scatter(self):
        torch.distributed.reduce_scatter_tensor(self._buffer, self._reduce_buffer)

    def _run_my_ring_all_reduce(self):
        # Initially, we want to write our own ring all reduce (https://github.com/NVIDIA/nccl/blob/master/src/device/all_reduce.h),
        # but soon we find it is too slow to invoke so many send receive functions from python.
        #
        # So, here we emulate 2 serialized encryptions, and 2 serialized decryptions. Note that since the encryption
        # and decryption can be done in parallel in ring all reduce, we also do them in parallel. We also have a memory copy
        # that copies the tensor from secure memory into the non-secure memory region so that NVLink can directly access the
        # encrypted tensor in the non-secure memory region (Refer to Nvidia H100 whitepaper).
        #
        # TODO: Need to learn more about NCCL internals for further optimization and a more reasonable encrypted baseline...
        # If uni-directional NVLink is 300GBps, bidirectional is 600GBps, and the encryption is 200GBps, then the overall BW is
        # 150GBps (4x slower).
        # TODO: should be 128-byte cacheline aligned !!! For each warp (32 threads)

        # Use all possible GPU resources for encryption in parallel
        blength = self._buffer.shape[0]
        bsplit_index = (self._nw - 1) * blength // self._nw
        pclength = self._pt.shape[0] // 2
        pcsplit_index = (self._nw - 1) * pclength // self._nw
        for i in range(2):  # This is the overall encrypted data of ring-all-reduce (throughput-oriented), 2(ngpu - 1)(sz/ngpu)
            torch.ops.extension_cpp.my_paralell_aes_encrypt(self._buffer[:bsplit_index].view(torch.uint32), self._ct[:pcsplit_index], self._iv, self._rek) # encryption
            torch.ops.extension_cpp.my_paralell_aes_encrypt(self._buffer[:bsplit_index].view(torch.uint32), self._pt[:pcsplit_index], self._iv, self._rek) # decryption (should be parallel to encryption ideally, but since each cuda operator can easily saturate SMs, it doesn't matter so much)

        # In each step, each GPU operates on total/ngpu data for integrity protection, and there are 2(ngpu - 1) rounds. In each round, the signature and authentication should be parallel.
        nblock = 1
        signum_in_uint32 = (self._ct.numel() // 2) // nblock
        nblock_for_all_comm = nblock * (self._nw - 1)   # There are 2(ngpu - 1) comm that cannot run in parallel
        for i in range(2):
            torch.ops.extension_cpp.my_paralell_gf128mul(self._ct[:pcsplit_index].view(torch.uint64), self._h, self._ct[self._ct.numel() // 2:].view(torch.uint64), nblock_for_all_comm)  # sign
            torch.ops.extension_cpp.my_paralell_gf128mul(self._pt[:pcsplit_index].view(torch.uint64), self._h, self._pt[self._pt.numel() // 2:].view(torch.uint64), nblock_for_all_comm)  # auth

        torch.distributed.all_reduce(self._ct[:(self._ct.numel() // 2) - 1 + signum_in_uint32].view(self._buffer.dtype))    # a tradeoff between par and NVLink consumption

    def _get_collective_fn(self, collective: str) -> Callable:
        if collective == "all_reduce":
            return self._run_all_reduce
        elif collective == "all_gather":
            return self._run_all_gather
        elif collective == "broadcast":
            return self._run_broadcast
        elif collective == "send_recv":
            return self._run_send_recv
        elif collective == "reduce_scatter":
            return self._run_reduce_scatter
        elif collective == "my_ring_all_reduce":
            return self._run_my_ring_all_reduce
        else:
            raise ValueError(f"Unknown collective: {collective}")

    def _build_graph(self) -> torch.cuda.CUDAGraph:
        # Warm up.
        for _ in range(WARMUP_STEPS):
            self._collective_fn()

        torch.cuda.synchronize()

        # Build graph.
        graph = torch.cuda.CUDAGraph()

        mempool = torch.cuda.graph_pool_handle()

        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
        ):
            with torch.cuda.graph(graph, mempool):
                for _ in range(GRAPH_STEPS):
                    self._collective_fn()

        torch.cuda.synchronize()
        return graph

    def launch(self) -> torch.Tensor:
        # NOTE: x must be a slice of self._buffer.
        if self._disable_graph:
            self._collective_fn()
        else:
            self._graph.replay()
