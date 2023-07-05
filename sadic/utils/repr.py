r"""Class for pretty printing of objects."""

import json


class Repr:
    r"""Class for pretty printing of objects."""

    def __repr__(self) -> str:
        r"""Pretty print the object."""
        return json.dumps(self.__dict__, indent=4, sort_keys=True, default=str)
