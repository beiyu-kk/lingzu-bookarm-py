"""Helpers for loading pyserial safely."""


def load_pyserial():
    """Import pyserial's ``serial`` module and reject unrelated ``serial`` packages.

    The project needs the ``Serial`` class and timeout exceptions provided by
    pyserial. Some environments also ship an unrelated package named
    ``serial``; that package does not expose the API we need.
    """

    try:
        import serial as serial_mod
    except ImportError as exc:
        raise ImportError(
            "pyserial is required. Install it with: pip install pyserial"
        ) from exc

    if not hasattr(serial_mod, "Serial"):
        module_file = getattr(serial_mod, "__file__", "<unknown>")
        raise ImportError(
            "The installed 'serial' package is not pyserial "
            f"({module_file}). Remove the unrelated 'serial' package and "
            "install pyserial instead: pip uninstall serial && pip install pyserial"
        )

    return serial_mod
