"""The registry is the contract between the instrumentation and the diagram.

`src/graphrag/trace/steps.py` declares the blocks; the code calls `step()` with
those names; `GET /trace/pipelines` serves nodes and edges built from the same
table. If those three can drift, the page shows a picture of a pipeline that no
longer exists — a block that silently stopped lighting up looks identical to a
block that never runs.

So this walks the source for every `step(...)` call and holds it to the table.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from graphrag.trace import steps as registry

_SRC = Path(__file__).resolve().parents[2] / "src" / "graphrag"


def _instrumented_names() -> set[str]:
    """Every name passed as the first argument to a `step(...)` call in `src/`.

    Read from the AST rather than by import, so a module that needs a live
    provider to import is still covered. Only `trace_steps.NAME` /
    `registry.NAME` attribute references are collected: a computed name could
    not be checked against the table anyway, and `test_every_step_call_names_a_constant`
    below is what stops one being introduced.
    """
    found: set[str] = set()
    for path in _SRC.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and _is_step_call(node.func)):
                continue
            if not node.args:
                continue
            first = node.args[0]
            if (
                isinstance(first, ast.Attribute)
                and isinstance(first.value, ast.Name)
                and first.value.id in ("trace_steps", "registry", "steps")
            ):
                found.add(first.attr)
    return found


def _referenced_names() -> set[str]:
    """Every `trace_steps.NAME` the source refers to, wherever it appears.

    Looser than `_instrumented_names` on purpose. Some call sites pass the name
    through a variable — `HybridRetriever` names its three legs in a tuple and
    hands each to a helper — so a scan that only reads `step()`'s first argument
    would report them as never emitted when they plainly are.
    """
    found: set[str] = set()
    for path in _SRC.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id in ("trace_steps", "registry", "steps")
            ):
                found.add(node.attr)
    return found


def _is_step_call(func: ast.expr) -> bool:
    if isinstance(func, ast.Name):
        return func.id == "step"
    if isinstance(func, ast.Attribute):
        return func.attr in ("step", "_step")
    return False


def test_every_instrumented_step_is_declared():
    """A `step()` call naming something the registry does not know is a block
    the diagram cannot draw."""
    undeclared = {
        name for name in _instrumented_names() if getattr(registry, name, None) is None
    }
    assert not undeclared, f"not in trace/steps.py: {sorted(undeclared)}"


def test_every_instrumented_step_resolves_to_a_spec():
    unspecced = sorted(
        name
        for name in _instrumented_names()
        if getattr(registry, name) not in registry.SPECS
    )
    assert not unspecced, f"declared but has no StepSpec: {unspecced}"


def test_every_declared_step_is_actually_instrumented():
    """The other direction: a node on the diagram that no code emits would sit
    there for ever looking like a stage that never runs."""
    emitted = {
        getattr(registry, n)
        for n in _referenced_names()
        if isinstance(getattr(registry, n, None), str)
    }
    # TRUNCATED is emitted by the recorder itself, not through `step()`.
    declared = set(registry.SPECS) - {registry.TRUNCATED}
    assert not (declared - emitted), f"drawn but never emitted: {sorted(declared - emitted)}"


@pytest.mark.parametrize("pipeline", registry.PIPELINES)
def test_edges_only_reference_nodes_of_their_own_pipeline(pipeline):
    graph = registry.pipeline_graph(pipeline)
    names = {n["name"] for n in graph["nodes"]}
    dangling = [
        e for e in graph["edges"] if e["from"] not in names or e["to"] not in names
    ]
    assert not dangling, f"{pipeline}: {dangling}"


@pytest.mark.parametrize("pipeline", registry.PIPELINES)
def test_every_pipeline_draws_something(pipeline):
    graph = registry.pipeline_graph(pipeline)
    assert graph["nodes"]
    assert graph["edges"]


def test_step_names_are_unique():
    assert len(registry.SPECS) == len(registry._SPECS)
