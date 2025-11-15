# src/trackers/pid_controller.py
import time

class PID:
    def __init__(self, Kp=0.05, Ki=0.0, Kd=0.01):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self._last = 0.0
        self._int = 0.0
        self._last_time = None

    def compute(self, error):
        now = time.time()
        if self._last_time is None:
            dt = 0.0
        else:
            dt = now - self._last_time
        self._last_time = now

        self._int += error * dt if dt>0 else 0.0
        deriv = (error - self._last)/dt if dt>0 else 0.0
        out = self.Kp*error + self.Ki*self._int + self.Kd*deriv
        self._last = error
        return out
