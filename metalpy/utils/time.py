import time


class Timer:
    def __init__(self):
        self.started = 0
        self.elapsed = 0
        pass

    def __enter__(self):
        self.started = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stopped = time.time()
        self.elapsed = self.stopped - self.started

    def __str__(self):
        elapsed_mins, elapsed_secs = epoch_time(self.started, self.stopped)
        if elapsed_mins == 0:
            return f"{elapsed_secs} s"
        else:
            return f"{elapsed_mins} min {elapsed_secs} s"


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs