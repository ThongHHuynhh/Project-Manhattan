"""Conveyor belt encoder: the only way the cutting stack learns where the belt is.

`SimBeltEncoder` integrates a simulated belt and reports what a real incremental encoder card would:
an unsigned tick count that wraps at 2**bits, read with a timestamp. `BeltTracker` turns readings into
belt position and speed. To use real hardware, implement `read()` on another class (e.g. a PLC/OPC UA
counter) with the same return type; nothing else changes.
"""
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class EncoderReading:
    ticks: int  # raw counter value, 0 .. 2**bits - 1 (wraps)
    t: float  # seconds (monotonic)


def tick_delta(new, old, bits=32):
    """Signed tick difference that survives counter wrap-around."""
    span = 1 << bits
    d = (new - old) % span
    return d - span if d >= span // 2 else d


class SimBeltEncoder:
    """Simulated incremental encoder on the belt drive."""

    def __init__(self, counts_per_m=10000.0, bits=32, start_ticks=0, clock=None):
        self.counts_per_m = float(counts_per_m)
        self.bits = bits
        self._pos_m = 0.0
        self._start = int(start_ticks)
        self._clock = clock or time.monotonic

    def advance(self, metres):
        """Move the simulated belt (call from the simulation step)."""
        self._pos_m += float(metres)

    def read(self) -> EncoderReading:
        ticks = (self._start + int(self._pos_m * self.counts_per_m)) % (1 << self.bits)
        return EncoderReading(ticks, self._clock())


class BeltTracker:
    """Belt position (m, since `zero`) and smoothed speed (m/s) from encoder readings."""

    def __init__(self, counts_per_m, bits=32, smoothing=0.2):
        self.counts_per_m, self.bits, self.smoothing = float(counts_per_m), bits, smoothing
        self._zero = self._last = None
        self.position = 0.0
        self.speed = 0.0

    def update(self, reading: EncoderReading):
        if self._zero is None:
            self._zero = self._last = reading
            return self.position, self.speed
        self.position = tick_delta(reading.ticks, self._zero.ticks, self.bits) / self.counts_per_m
        dt = reading.t - self._last.t
        if dt > 0:
            v = tick_delta(reading.ticks, self._last.ticks, self.bits) / self.counts_per_m / dt
            self.speed += self.smoothing * (v - self.speed)
        self._last = reading
        return self.position, self.speed

    def distance_since(self, latch_ticks, reading: EncoderReading):
        """Metres the belt moved since a latched count (e.g. the tray's photoeye trigger)."""
        return tick_delta(reading.ticks, latch_ticks, self.bits) / self.counts_per_m
