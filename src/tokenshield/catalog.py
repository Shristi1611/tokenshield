"""Ready-made schemas and policies for the content types the gateway understands.

README_FACTS (in schema.py) covers README-like files. LOG_FACTS covers build and test output,
the most common large tool result in coding-agent sessions. Anything with no schema is blocked.
"""

from __future__ import annotations

from tokenshield.gate import GatePolicy, SourceKind
from tokenshield.schema import README_FACTS, FactSchema, FieldSpec, FieldType

LOG_FACTS = FactSchema(
    name="log_facts",
    fields=(
        FieldSpec("build_ok", FieldType.BOOL),
        FieldSpec(
            "tests_passed",
            FieldType.INT,
            min_value=0,
            max_value=1_000_000,
            hint="Use 0 when the log shows no test results.",
        ),
        FieldSpec(
            "tests_failed",
            FieldType.INT,
            min_value=0,
            max_value=1_000_000,
            hint="Use 0 when the log shows no test results.",
        ),
        FieldSpec("num_warnings", FieldType.INT, min_value=0, max_value=1_000_000),
        FieldSpec(
            "error_kind",
            FieldType.ENUM,
            choices=(
                "none",
                "syntax",
                "import",
                "assertion",
                "timeout",
                "permission",
                "network",
                "other",
            ),
            hint="Pick the main kind of error, or none when the run succeeded.",
        ),
    ),
)


def gateway_policy() -> GatePolicy:
    """Files are README-like and tool results are build or test logs. Other kinds are blocked."""
    return GatePolicy({SourceKind.FILE: README_FACTS, SourceKind.TOOL_OUTPUT: LOG_FACTS})