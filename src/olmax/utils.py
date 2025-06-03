def mib_to_bytes(mb: float) -> int:
    return int(1024 * 1024 * mb)


def bytes_to_mib(b: int) -> float:
    return b / (1024 * 1024)
