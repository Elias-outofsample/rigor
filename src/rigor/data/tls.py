"""TLS handling for machines behind an SSL-inspection layer (corporate AV/proxy).

Rigor machines sit behind an SSL-inspection layer that presents a non-RFC-compliant
root cert, which OpenSSL (and therefore ``requests``/``urllib``) rejects. The
correct fix -- used framework-wide -- is ``truststore``, which delegates
verification to the operating-system trust store (where that root *is* trusted),
keeping verification ON. We never disable TLS verification.

Call :func:`enable_tls` once before any network access.
"""

from __future__ import annotations

_INJECTED = False


def enable_tls() -> None:
    """Route TLS verification through the OS trust store (idempotent)."""
    global _INJECTED
    if _INJECTED:
        return
    import truststore

    truststore.inject_into_ssl()
    _INJECTED = True
