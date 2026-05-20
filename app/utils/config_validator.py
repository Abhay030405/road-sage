"""Config validation for RoadSage YAML configuration files.

Usage (CLI):
    python app/utils/config_validator.py configs/production.yaml
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any

import yaml

# Sentinel distinct from None so _fail can tell apart "key missing" vs "key is None".
_MISSING = object()


@dataclass
class ConfigValidationError(Exception):
    """Raised when a RoadSage config value is absent, the wrong type, or violates a constraint.

    Attributes:
        key_path: Dot-notation path to the offending key (e.g. ``"models.lane_detector.path"``).
        expected_type: Human-readable description of the expected type or constraint.
        got: Human-readable description of the actual value that was found.
        message: Full error message suitable for display to the user.
    """

    key_path: str
    expected_type: str
    got: str
    message: str

    def __str__(self) -> str:
        return self.message


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_nested(config: dict, key_path: str) -> tuple[bool, Any]:
    """Traverse *config* using a dot-notation *key_path*.

    Args:
        config: The top-level config dictionary.
        key_path: Dot-separated key path, e.g. ``"models.lane_detector.path"``.

    Returns:
        A ``(found, value)`` tuple.  *found* is ``False`` when any segment of
        the path is absent or the intermediate node is not a dict.
    """
    node: Any = config
    for part in key_path.split("."):
        if not isinstance(node, dict) or part not in node:
            return False, None
        node = node[part]
    return True, node


def _fail(
    key_path: str,
    expected_type: str,
    value: Any,
    detail: str = "",
) -> None:
    """Raise :class:`ConfigValidationError` with a consistent message format.

    Args:
        key_path: Dot-notation path to the offending config key.
        expected_type: Description of what was expected (e.g. ``"float in [0, 1]"``).
        value: The actual value found, or the ``_MISSING`` sentinel.
        detail: Optional extra clause appended after an em-dash (e.g. a range message).

    Raises:
        ConfigValidationError: Always.
    """
    got_str = "missing" if value is _MISSING else f"{type(value).__name__}({value!r})"
    msg = f"Config key '{key_path}': expected {expected_type}, got {got_str}"
    if detail:
        msg += f" — {detail}"
    raise ConfigValidationError(
        key_path=key_path,
        expected_type=expected_type,
        got=got_str,
        message=msg,
    )


def _get(config: dict, key_path: str, expected_type: str) -> Any:
    """Return the value at *key_path* or immediately raise if the key is absent."""
    found, value = _get_nested(config, key_path)
    if not found:
        _fail(key_path, expected_type, _MISSING)
    return value


# ---------------------------------------------------------------------------
# Type-specific validators (closures created inside validate_config for brevity)
# ---------------------------------------------------------------------------

def _make_validators(config: dict):
    """Return a namespace of validator callables bound to *config*.

    Each validator fetches the value, checks its type and optional constraints,
    and raises :class:`ConfigValidationError` on the first violation.
    """

    def require_str(kp: str) -> str:
        v = _get(config, kp, "str")
        if not isinstance(v, str):
            _fail(kp, "str", v)
        return v

    def require_bool(kp: str) -> bool:
        v = _get(config, kp, "bool")
        if not isinstance(v, bool):
            _fail(kp, "bool", v)
        return v

    def require_int(kp: str) -> int:
        """Strict int — rejects bools even though bool is an int subtype."""
        v = _get(config, kp, "int")
        if isinstance(v, bool) or not isinstance(v, int):
            _fail(kp, "int", v)
        return v

    def require_float(
        kp: str,
        *,
        lo: float | None = None,
        hi: float | None = None,
    ) -> float:
        """Float (or int promoted to float) with optional inclusive bounds."""
        v = _get(config, kp, "float")
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            _fail(kp, "float", v)
        v = float(v)
        if lo is not None and v < lo:
            _fail(kp, f"float >= {lo}", v, f"{v} is below minimum {lo}")
        if hi is not None and v > hi:
            _fail(kp, f"float <= {hi}", v, f"{v} exceeds maximum {hi}")
        return v

    def require_number(kp: str) -> int | float:
        """Accept either int or float (rejects bool)."""
        v = _get(config, kp, "int or float")
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            _fail(kp, "int or float", v)
        return v

    def require_str_choice(kp: str, choices: list[str]) -> str:
        v = require_str(kp)
        if v not in choices:
            _fail(kp, f"one of {choices}", v, f"'{v}' is not an allowed value")
        return v

    def require_point_list(kp: str, length: int) -> list:
        """Validate that the value is a list with exactly *length* elements."""
        v = _get(config, kp, f"list of {length} points")
        if not isinstance(v, list):
            _fail(kp, f"list of {length} points", v)
        if len(v) != length:
            _fail(
                kp,
                f"list of {length} points",
                v,
                f"got {len(v)} point(s) instead of {length}",
            )
        return v

    return (
        require_str,
        require_bool,
        require_int,
        require_float,
        require_number,
        require_str_choice,
        require_point_list,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_config(config: dict) -> None:
    """Validate a loaded RoadSage config dictionary against required keys and types.

    Checks that every required key is present, has the correct type, and (where
    applicable) satisfies value constraints such as numeric ranges or enumerated
    choices.  Validation stops at the first violation.

    Args:
        config: The configuration dictionary, typically produced by ``yaml.safe_load``.

    Raises:
        ConfigValidationError: When a required key is missing, has an unexpected
            type, or its value fails a constraint (e.g. a confidence threshold
            outside ``[0, 1]``).
    """
    (
        require_str,
        require_bool,
        require_int,
        require_float,
        require_number,
        require_str_choice,
        require_point_list,
    ) = _make_validators(config)

    # project
    require_str("project.name")
    require_str("project.version")

    # device
    require_str_choice("device.type", ["cpu", "cuda"])
    require_bool("device.use_onnx")

    # models
    require_str("models.lane_detector.path")
    require_float("models.lane_detector.confidence_threshold", lo=0.0, hi=1.0)
    require_float("models.object_detector.confidence_threshold", lo=0.0, hi=1.0)
    require_str("models.depth_estimator.path")
    require_str("models.decision_cnn.path")

    # decision_engine
    require_float("decision_engine.offset_threshold")
    require_float("decision_engine.curve_threshold")
    require_float("decision_engine.obstacle_stop_distance")
    require_float("decision_engine.min_confidence", lo=0.0, hi=1.0)

    # quality_filters
    require_number("quality_filters.blur_laplacian_threshold")
    require_int("quality_filters.brightness_min")
    require_int("quality_filters.brightness_max")
    require_float("quality_filters.road_coverage_min")

    # bev_transform
    require_point_list("bev_transform.src_points", 4)
    require_point_list("bev_transform.dst_points", 4)
    require_float("bev_transform.pixels_per_meter")


def load_and_validate_config(config_path: str) -> dict:
    """Load a YAML config file from disk and validate its contents.

    Reads the file at *config_path*, parses it with ``yaml.safe_load``, and
    passes the result to :func:`validate_config`.

    Args:
        config_path: Path to the YAML configuration file (e.g.
            ``"configs/production.yaml"``).

    Returns:
        The validated configuration dictionary, ready for use by the application.

    Raises:
        ConfigValidationError: If the file cannot be found, is not valid YAML,
            the top-level document is not a mapping, or any required key fails
            validation.
    """
    try:
        with open(config_path, encoding="utf-8") as fh:
            config = yaml.safe_load(fh)
    except FileNotFoundError:
        raise ConfigValidationError(
            key_path="<file>",
            expected_type="readable YAML file",
            got=f"no file at '{config_path}'",
            message=f"Config file not found: '{config_path}'",
        )
    except yaml.YAMLError as exc:
        raise ConfigValidationError(
            key_path="<file>",
            expected_type="valid YAML",
            got="parse error",
            message=f"Failed to parse '{config_path}': {exc}",
        )

    if not isinstance(config, dict):
        raise ConfigValidationError(
            key_path="<root>",
            expected_type="YAML mapping",
            got=type(config).__name__,
            message=(
                f"Top-level document in '{config_path}' must be a YAML mapping, "
                f"got {type(config).__name__}"
            ),
        )

    validate_config(config)
    return config


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python app/utils/config_validator.py <config_path>")
        sys.exit(1)

    try:
        load_and_validate_config(sys.argv[1])
        print("Config valid.")
    except ConfigValidationError as exc:
        print(f"Validation error: {exc}")
        sys.exit(1)
