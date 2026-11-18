"""Tiny test runner for the TODO cells in the notebooks.

``check("name", test_function)`` runs a test function and prints one line:

* ``PASS``  - the test function ran without complaint,
* ``TODO``  - your function still raises ``NotImplementedError``,
* ``FAIL``  - an ``assert`` in the test failed (the message tells you why),
* ``ERROR`` - your function crashed; the traceback is printed below.

The notebook keeps running in every case, so you can solve the tasks in any
order.
"""

from __future__ import annotations

import traceback
from collections.abc import Callable


def check(name: str, test: Callable[[], None]) -> None:
    """Run ``test()`` and print one line: PASS / TODO / FAIL / ERROR."""
    try:
        test()
    except NotImplementedError as err:
        print(f"TODO   {name}: not implemented yet ({err})")
        return
    except AssertionError as err:
        print(f"FAIL   {name}: {err}")
        return
    except Exception:  # noqa: BLE001 - we want to show every crash to the student
        print(f"ERROR  {name}: your code raised an exception:")
        traceback.print_exc()
        return
    print(f"PASS   {name}")
