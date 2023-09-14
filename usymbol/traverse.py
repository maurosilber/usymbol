from functools import singledispatch

from frozendict import frozendict


@singledispatch
def traverse(obj, apply):
    """Traverses an object applying a function to its elements."""
    return apply(obj)


@traverse.register
def traverse_tuple(obj: tuple, apply):
    return apply(tuple(traverse(o, apply) for o in obj))


@traverse.register
def traverse_frozendict(obj: frozendict, apply):
    return apply(frozendict({k: traverse(v, apply) for k, v in obj.items()}))
