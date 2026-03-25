"""Index wrapper."""


class Index:
    def __init__(self, native_index):
        self._native = native_index

    def __len__(self):
        return len(self._native)

    def __repr__(self):
        return repr(self._native)

    def tolist(self):
        return self._native.tolist()
