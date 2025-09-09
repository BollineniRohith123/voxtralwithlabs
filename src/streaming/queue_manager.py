import asyncio

class QueueManager:
    def __init__(self, config):
        self.config = config
        self._queue = asyncio.Queue()

    def get_queue_size(self) -> int:
        return self._queue.qsize()
