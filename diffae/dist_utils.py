from torch import distributed


def barrier() -> None:
    if distributed.is_initialized():
        distributed.barrier()
    else:
        pass


def broadcast(data, src) -> None:
    if distributed.is_initialized():
        distributed.broadcast(data, src)
    else:
        pass


def all_gather(data: list, src) -> None:
    if distributed.is_initialized():
        distributed.all_gather(data, src)
    else:
        data[0] = src


def get_rank() -> int:
    if distributed.is_initialized():
        return distributed.get_rank()
    else:
        return 0


def get_world_size() -> int:
    if distributed.is_initialized():
        return distributed.get_world_size()
    else:
        return 1


def chunk_size(size: int, rank: int, world_size: int) -> int:
    extra = rank < size % world_size
    return size // world_size + extra
