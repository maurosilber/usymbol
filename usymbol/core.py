from __future__ import annotations

from collections import Counter, defaultdict
from typing import Callable, Container, Mapping, Sequence

from .symbol import Call, Symbol
from .traverse import traverse


def substitute(expr: Symbol, mapper: Callable | Mapping) -> Symbol:
    """Traverses the expression and substitutes every node according to mapper.

    If mapper is a Callable, every element is applied mapper (`mapper(expr)`).
    If mapper is a Mapping, every element is searched for in the mapping or left as is.
    """
    if isinstance(mapper, Mapping):
        return traverse(expr, lambda x: mapper.get(x, x))
    else:
        return traverse(expr, mapper)


def evaluate(expr: Symbol) -> Symbol:
    """Traverses the expression evaluating every Call instance."""

    def evaluator(result):
        if isinstance(result, Call):
            return result.__call__()
        else:
            return result

    return traverse(expr, evaluator)


def substitute_and_evaluate(expr, mapper: Callable | Mapping):
    """Traverses the expression evaluating every Call instance.

    Faster variant than evaluate(substitute(expr)) as it only traverses the expression once.
    """
    if isinstance(mapper, Mapping):

        def normalized_mapper(x):
            return mapper.get(x, x)

    else:
        normalized_mapper = mapper

    def evaluator(expr):
        result = normalized_mapper(expr)

        if isinstance(result, Call):
            return result.__call__()
        else:
            return result

    return traverse(expr, evaluator)


def inspect(expr: Symbol) -> Counter:
    """Traverses the expression collecting every node into a Counter."""
    counter = Counter()

    def count(x):
        counter[x] += 1
        return x

    traverse(expr, count)
    return counter


def to_function(
    expr: Symbol,
    parameters: Sequence[Symbol],
    *,
    name: str,
) -> Callable:
    # Find:
    # - all functions used in Calls,
    #   that we will need to provide in the globals dict,
    # - all Symbols in the expression,
    #   which will be replaced by their string representation.
    # Then:
    #   - check for collisions in symbol_names,
    #   - generate unique names for the functions in the globals dict,
    #   - replace the functions for these names in the expression.
    symbol_names = defaultdict[str, set[Symbol]](set)
    functions = set[Callable]()

    def inspect(x):
        if isinstance(x, Call):
            functions.add(x.func)
        else:
            symbol_names[str(x)].add(x)
        return x  # return as is, we don't want to change the expression yet

    def generate_unique_names(
        used_names: Container[str],
        requested_names: int,
    ) -> set[str]:
        names = set()
        i = 0
        while len(names) < requested_names:
            proposed_name = f"f{i}"
            if proposed_name not in used_names:
                names.add(proposed_name)
            i += 1
        return names

    traverse(expr, inspect)  # Find

    collisions = {k: v for k, v in symbol_names.items() if len(v) > 1}
    if len(collisions) > 0:
        raise ValueError("More than one Symbol maps to the same name", collisions)

    unique_names = generate_unique_names(
        used_names=symbol_names,
        requested_names=len(functions),
    )
    expr = substitute(expr, dict(zip(functions, unique_names)))  # Replace
    globals = dict(zip(unique_names, functions))  # inverse mapping

    # Compile and return the function
    sig = ",".join(map(str, parameters))
    func_def = "\n\t".join(
        [
            f"def {name}({sig}):",
            f"return {expr}",
        ]
    )
    locals = {}
    exec(func_def, globals, locals)
    return locals[name]
