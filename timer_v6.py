import time
import warnings
import threading
from threading import Thread

"""
这一版的主要改动是把景点计时器放到主线程里面，
这样做的好处是游客在游览景点的时候，线程是阻塞的，
那么这样就不会让循环程序在游客游览经典的时候对动作空间采样，
之前的版本会产生很大的噪声，因为在游客游览的时候，不管执行什么动作，他的回报都是负的。
"""

def countdown(timer, elapsed):
    if timer._start is None:
        timer.start()
        t=timer._func() - timer._start
        while t < elapsed:
            t=timer._func() - timer._start
        timer.elapsed=timer.elapsed-t
        timer.stop()
        timer.finished=True
        return True
    else:
        timer.finished=False
        return False

class countdown_with_return(Thread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs=None, *, daemon=None):
        Thread.__init__(self, group, target, name, args, kwargs, daemon=daemon)
        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)

    def join(self):
        Thread.join(self)
        return self._return

class Timer:
    def __init__(self, elapsed, func=time.perf_counter):
        self.elapsed = elapsed
        self._func = func
        self._start = None
        self.finished = 0
        countThread = threading.Thread(target=countdown, args=(self, elapsed))
        # countThread = countdown_with_return(target=countdown, args=(self, elapsed))
        countThread.start()

    def countdown(self, elapsed):
        if self._start is None:
            self.start()
            t=self._func() - self._start
            while t < elapsed:
                t=self._func() - self._start
            self.elapsed-=t
            self.stop()
            return True
        else:
            return False

    def getElapsed(self):
        if self._start is not None:
            # 有可能在if检查的时候self._start 还不是 None，但是到了执行下面的return语句的时候就是None了
            try:
                return self.elapsed-(self._func() - self._start)
            finally:
                return self.elapsed
        else:
            return self.elapsed

    def start(self):
        if self._start is not None:
            warnings.warn('Already started')
        self._start = self._func()

    def stop(self):
        if self._start is None:
            warnings.warn('Not started')
        self._start = None

    def reset(self):
        self.elapsed = 0.0

    @property
    def running(self):
        return self._start is not None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

class SpotTimer:
    def __init__(self, elapsed, func=time.perf_counter):
        self.elapsed = elapsed
        self._func = func
        self._start = None
        self.finished = 0
        self.countdown(elapsed)

    def countdown(self, elapsed):
        if self._start is None:
            self.start()
            t=self._func() - self._start
            while t < elapsed:
                t=self._func() - self._start
            self.elapsed-=t
            self.stop()
            return True
        else:
            return False

    def getElapsed(self):
        if self._start is not None:
            # 有可能在if检查的时候self._start 还不是 None，但是到了执行下面的return语句的时候就是None了
            try:
                return self.elapsed-(self._func() - self._start)
            finally:
                return self.elapsed
        else:
            return self.elapsed

    def start(self):
        if self._start is not None:
            warnings.warn('Already started')
        self._start = self._func()

    def stop(self):
        if self._start is None:
            warnings.warn('Not started')
        self._start = None

    def reset(self):
        self.elapsed = 0.0

    @property
    def running(self):
        return self._start is not None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

class Timer_v6:

    def __init__(self, elapsed):
        self.elapsed=elapsed
        self.startTime=time.perf_counter()
        self.nextFinish=self.startTime

    def useTime(self, timeCost=0):
        self.nextFinish+=timeCost

    def checkAvailability(self):
        if self.nextFinish-self.startTime<self.elapsed:
            return True 
        else:
            return False
    def getElapsed(self):
        return self.elapsed-(self.nextFinish-self.startTime)


