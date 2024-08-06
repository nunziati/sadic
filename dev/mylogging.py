
class TaskPrinter:
    def __init__(self, verbose=True, length=60):
        self.verbose = verbose
        self.length = length
        self.active_task = False

    def print_task(self, task):
        if self.verbose:
            print(task.ljust(self.length, "."), end="", flush=True)

    def print_task_done(self):
        if self.verbose:
            print("DONE", flush=True)

    def __call__(self, task = None):
        if task is None:
            if not self.active_task:
                raise ValueError("No active task")
            self.print_task_done()
            self.active_task = False

        elif not self.active_task:
            self.print_task(task)
            self.active_task = True
        else:
            self.print_task_done()
            self.active_task = False
            self(task)