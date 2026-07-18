"""Training lifecycle package.

Callers import the explicit ``runtime``, ``trainer`` or ``checkpoint`` module.
Keeping this initializer lazy prevents evaluation/checkpoint imports from loading
the full trainer and constructing a package-level dependency cycle.
"""

__all__: tuple[str, ...] = ()
