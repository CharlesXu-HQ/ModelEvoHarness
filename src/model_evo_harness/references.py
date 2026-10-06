"""Host-observed source reads and delivery to a completion callback.

This is an audit boundary for model output, not isolation from Python adapters.
Reading or delivering source does not prove that a candidate uses it correctly.
"""

from __future__ import annotations

import hashlib
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy


_active_reads: ContextVar[dict | None] = ContextVar("model_evo_reference_reads", default=None)


@contextmanager
def _collect_reference_events():
    ledger = {"reads": {}, "events": []}
    token = _active_reads.set(ledger)
    try:
        yield ledger
    finally:
        _active_reads.reset(token)


def _record_reference_read(files: dict) -> None:
    ledger = _active_reads.get()
    if ledger is not None:
        hashes = {path: entry["sha256"] for path, entry in files.items()}
        ledger["reads"].update(hashes)
        ledger["events"].append({"event": "read", "files": hashes})


def call_with_references(complete, context: dict):
    """Supply source material to a host completion callback and record delivery.

    Custom adapters call ``read_references`` inside ``propose`` and place the
    returned ``files`` in ``context.reference_material``. In ``run_search``,
    delivery must match an actual read in that proposal scope. The callback is
    responsible for submitting this context to its model. A callback invocation
    is recorded even if the provider fails; it is not a model-use attestation.
    """
    delivered = deepcopy(context)
    material = delivered.get("reference_material", {})
    ledger = _active_reads.get()
    if ledger is not None and material:
        hashes = {}
        for path, entry in material.items():
            content = entry.get("content") if isinstance(entry, dict) else None
            digest = (hashlib.sha256(content.encode()).hexdigest()
                      if isinstance(content, str) else None)
            if (digest is None or ledger["reads"].get(path) != digest or
                    entry.get("sha256") != digest):
                raise ValueError("reference delivery needs a matching host-observed read")
            hashes[path] = digest
        ledger["events"].append({"event": "delivered", "files": hashes})
    return complete(delivered)
