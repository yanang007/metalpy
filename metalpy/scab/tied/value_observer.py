import threading
import time


class ValueObserver(threading.Thread):
    def __init__(self, value_supplier, callback, interval=0.5):
        threading.Thread.__init__(self)
        self.value_supplier = value_supplier
        self.callback = callback
        self.interval = interval
        self.last_n = 0
        self.running = True

        self.lock = threading.Lock()

    def sync(self, n):
        with self.lock:
            self.update(n - self.last_n)

    def update(self, dx):
        self.callback(dx)
        self.last_n += dx

    def run(self):
        while self.running:
            time.sleep(self.interval)
            with self.lock:
                n = self.value_supplier()
                dx = n - self.last_n
                self.update(dx)

    def stop(self):
        self.running = False
