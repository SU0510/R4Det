# Minimal stub for detectron2.utils.events

import datetime
import logging

logger = logging.getLogger(__name__)

def get_event_storage():
    return _EventStorageStub()


class EventStorage(_EventStorageStub if '_EventStorageStub' in dir() else object):
    """Stub of detectron2.utils.events.EventStorage."""
    pass


class _EventStorageStub:
    """Silent stub that satisfies the interface but does nothing."""
    
    def __init__(self, start_iter=0):
        self._iter = start_iter
        self._history = {}
        self._current_prefix = ""

    def put_scalar(self, name, value, smoothing_hint=True):
        pass

    def put_scalars(self, *, scalar_dict=None, smoothing_hint=True):
        pass

    def history(self, name, window_size=20):
        return []

    def latest(self, name):
        return 0.0

    def smooth(self, name, alpha=0.9):
        return 0.0

    def put_image(self, name, tensor):
        pass

    def put_histogram(self, name, tensor, bins=None):
        pass

    def put_artifact(self, name, content, content_type):
        pass

    def iterations(self):
        return self._iter

    def step(self):
        self._iter += 1

    @property
    def iter(self):
        return self._iter

    @property
    def iteration(self):
        return self._iter

    def name_scope(self, name):
        return _DummyScope()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class _DummyScope:
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class EventWriter:
    """Stub of detectron2.utils.events.EventWriter (abstract base)."""
    def write(self):
        pass
    def close(self):
        pass


class CommonMetricPrinter(EventWriter):
    """Stub of detectron2.utils.events.CommonMetricPrinter."""
    def __init__(self, max_iter=None, window_size=20):
        self._max_iter = max_iter
        self._window_size = window_size
    def write(self):
        pass


class JSONWriter(EventWriter):
    """Stub of detectron2.utils.events.JSONWriter."""
    def __init__(self, json_file, window_size=20):
        self._file_handle = open(json_file, 'w') if isinstance(json_file, str) else json_file
    def write(self):
        pass
    def close(self):
        self._file_handle.close()


class TensorboardXWriter(EventWriter):
    """Stub of detectron2.utils.events.TensorboardXWriter."""
    def __init__(self, log_dir, window_size=20, **kwargs):
        pass
    def write(self):
        pass
    def close(self):
        pass


# Alias for compatibility
EventStorage = _EventStorageStub
