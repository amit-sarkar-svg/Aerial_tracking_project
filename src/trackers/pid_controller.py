import time

class PID:
    def __init__(self, kp, ki, kd):
        self.Kp = kp
        self.Ki = ki
        self.Kd = kd

        self.last_error = 0
        self.integral = 0
        self.last_time = None

    def compute(self, error):
        now = time.time()
        if self.last_time is None:
            dt = 0.01
        else:
            dt = now - self.last_time

        self.integral += error * dt
        derivative = (error - self.last_error) / dt

        output = (
            self.Kp * error
            + self.Ki * self.integral
            + self.Kd * derivative
        )

        self.last_error = error
        self.last_time = now
        return output
