""" basic operators
"""

import time


class Operator:
    def __init__(self, op=None):
        self.ops = [op] if op is not None else list()

    def __add__(self, op2):
        assert isinstance(op2, Operator)
        op_list = self.ops.copy()
        op_list += op2.ops
        op_ret = Operator()
        op_ret.ops = op_list
        return op_ret


class Sleeper(Operator):
    def __init__(self, time_sleep=1.0):
        self.time_sleep = time_sleep
        super().__init__(self)

    def __call__(self):
        time.sleep(self.time_sleep)


class Printer(Operator):
    def __init__(self, words = "hello world!", end="\n"):
        self.words = words
        self.end = end
        super().__init__(self)
    
    def __call__(self):
        print(self.words, end=self.end)


def do(*ops, times=1):
    if len(ops) == 1:
        if isinstance(times, int) and times >= 0:
            for _ in range(times):
                for sub_op in ops[0].ops: sub_op()
        else:
            while True:
                for sub_op in ops[0].ops: sub_op()
    else:
        if isinstance(times, int) and times >= 0:
            for _ in range(times):
                for sub_op in ops: do(sub_op)
        else:
            while True:
                for sub_op in ops: do(sub_op)



if __name__ == "__main__":
    """ examples for calling ops
    """
    from operator import add
    from functools import reduce
    
    do(Printer("wait for 1s") + Sleeper(1.0) + 
       Printer("wait for 2s") + Sleeper(2.0) + 
       Printer("wait for 3s") + Sleeper(3.0) + 
       Printer("wait for 4s") + Sleeper(4.0)   )
    Printer("===========================")()

    do(reduce(add, [Printer("wait for 1s"), Sleeper(1.0)] * 4))
    Printer("===========================")()

    do(Printer("wait for 1s"), Sleeper(1.0), 
       Printer("wait for 2s"), Sleeper(2.0), )
    Printer("===========================")()

    do(*([Printer("wait for 1s"), Sleeper(1.0)] * 3))
    Printer("===========================")()

    do(Printer("wait for 1s"), Sleeper(1.0), times=5)
    Printer("===========================")()

    do(Printer("wait for 1s") + Sleeper(1.0) + Printer("wait for 1s"), Sleeper(1.0), times=3)
    Printer("===========================")()

