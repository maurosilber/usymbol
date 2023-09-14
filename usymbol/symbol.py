from __future__ import annotations

import inspect
import operator
from dataclasses import dataclass
from functools import reduce
from typing import Any, Callable, Generic, Mapping, ParamSpec, TypeVar

from frozendict import frozendict

from .traverse import traverse

P = ParamSpec("P")
R = TypeVar("R")


class Symbol:
    def __str__(self):
        raise NotImplementedError

    def __getitem__(self, item):
        return OperatorCall(operator.getitem, (self, item), op="{}[{}]", precedence=5)

    def __setitem__(self, item, value):
        return Call(operator.setitem, (self, item, value))

    # Unary operators
    def __pos__(self):
        return UnaryCall(operator.pos, (self,), op="+", precedence=2)

    def __neg__(self):
        return UnaryCall(operator.neg, (self,), op="-", precedence=2)

    def __invert__(self):
        return UnaryCall(operator.invert, (self,), op="~", precedence=2)

    # Comparison methods (ComparisonCall implements __bool__)
    def __eq__(self, other):
        return BinaryCall(operator.eq, (self, other), precedence=-5, op="==")

    def __ne__(self, other):
        return BinaryCall(operator.ne, (self, other), precedence=-5, op="!=")

    def __lt__(self, other):
        return BinaryCall(operator.lt, (self, other), precedence=-5, op="<")

    def __le__(self, other):
        return BinaryCall(operator.le, (self, other), precedence=-5, op="<=")

    def __gt__(self, other):
        return BinaryCall(operator.gt, (self, other), precedence=-5, op=">")

    def __ge__(self, other):
        return BinaryCall(operator.ge, (self, other), precedence=-5, op=">=")

    # Binary operators
    def __add__(self, other):
        return BinaryCall(operator.add, (self, other), precedence=0, op="+")

    def __sub__(self, other):
        return BinaryCall(operator.sub, (self, other), precedence=0, op="-")

    def __mul__(self, other):
        return BinaryCall(operator.mul, (self, other), precedence=1, op="*")

    def __matmul__(self, other):
        return BinaryCall(operator.matmul, (self, other), precedence=1, op="@")

    def __truediv__(self, other):
        return BinaryCall(operator.truediv, (self, other), precedence=1, op="/")

    def __floordiv__(self, other):
        return BinaryCall(operator.floordiv, (self, other), precedence=1, op="//")

    def __mod__(self, other):
        return BinaryCall(operator.mod, (self, other), precedence=1, op="%")

    def __pow__(self, other):
        return BinaryCall(operator.pow, (self, other), precedence=3, op="**")

    def __lshift__(self, other):
        return BinaryCall(operator.lshift, (self, other), precedence=-1, op="<<")

    def __rshift__(self, other):
        return BinaryCall(operator.rshift, (self, other), precedence=-1, op=">>")

    def __and__(self, other):
        return BinaryCall(operator.and_, (self, other), precedence=-2, op="&")

    def __xor__(self, other):
        return BinaryCall(operator.xor, (self, other), precedence=-3, op="^")

    def __or__(self, other):
        return BinaryCall(operator.or_, (self, other), precedence=-4, op="|")

    # Reversed binary operators
    def __radd__(self, other):
        return BinaryCall(operator.add, (other, self), precedence=0, op="+")

    def __rsub__(self, other):
        return BinaryCall(operator.sub, (other, self), precedence=0, op="-")

    def __rmul__(self, other):
        return BinaryCall(operator.mul, (other, self), precedence=1, op="*")

    def __rmatmul__(self, other):
        return BinaryCall(operator.matmul, (other, self), precedence=1, op="@")

    def __rtruediv__(self, other):
        return BinaryCall(operator.truediv, (other, self), precedence=1, op="/")

    def __rfloordiv__(self, other):
        return BinaryCall(operator.floordiv, (other, self), precedence=1, op="//")

    def __rmod__(self, other):
        return BinaryCall(operator.mod, (other, self), precedence=1, op="%")

    def __rpow__(self, other):
        return BinaryCall(operator.pow, (other, self), precedence=3, op="**")

    def __rlshift__(self, other):
        return BinaryCall(operator.lshift, (other, self), precedence=-1, op="<<")

    def __rrshift__(self, other):
        return BinaryCall(operator.rshift, (other, self), precedence=-1, op=">>")

    def __rand__(self, other):
        return BinaryCall(operator.and_, (other, self), precedence=-2, op="&")

    def __rxor__(self, other):
        return BinaryCall(operator.xor, (other, self), precedence=-3, op="^")

    def __ror__(self, other):
        return BinaryCall(operator.or_, (other, self), precedence=-4, op="|")


@dataclass(frozen=True)
class Call(Symbol, Generic[R]):
    func: Callable
    args: tuple = ()
    kwargs: Mapping = frozendict()

    def __post_init__(self):
        if not isinstance(self.kwargs, frozendict):
            object.__setattr__(self, "kwargs", frozendict(self.kwargs))

    def __str__(self):
        if isinstance(self.func, str):
            func = self.func
        else:
            try:
                func = f"{self.func.__module__}.{self.func.__qualname__}"
            except AttributeError:
                # This works for numpy functions
                func = f"{self.func.__class__.__module__}.{self.func.__name__}"
        args = ", ".join(map(str, self.args))
        return f"{func}({args})"

    def __call__(self):
        return self.func(*self.args, **self.kwargs)


@traverse.register
def traverse_call(obj: Call, apply):
    return apply(
        obj.__class__(
            func=traverse(obj.func, apply),
            args=traverse(obj.args, apply),
            kwargs=traverse(obj.kwargs, apply),
        )
    )


@dataclass(frozen=True, kw_only=True)
class OperatorCall(Call):
    op: str
    precedence: int

    def __str__(self):
        args = []
        for arg in self.args:
            if isinstance(arg, OperatorCall) and (
                (arg.precedence < self.precedence) or (arg.op == self.op)
            ):
                args.append(f"({arg})")
            else:
                args.append(str(arg))

        if "{}" in self.op:
            return reduce(self.op.format, args)
        else:
            return self.op.join(args)

    def __call__(self):
        if len(self.args) > 2:
            return reduce(self.func, self.args)
        return super().__call__()


@traverse.register
def traverse_operatorcall(obj: OperatorCall, apply):
    return apply(
        obj.__class__(
            func=traverse(obj.func, apply),
            args=traverse(obj.args, apply),
            kwargs=traverse(obj.kwargs, apply),
            op=obj.op,
            precedence=obj.precedence,
        )
    )


@dataclass(frozen=True)
class UnaryCall(OperatorCall):
    def __str__(self):
        return f"{self.op}{self.args[0]}"


@dataclass(frozen=True)
class BinaryCall(OperatorCall):
    def __post_init__(self):
        match self.args:
            case [BinaryCall() as left, right] if left.op is self.op:
                args = (*left.args, right)
                object.__setattr__(self, "args", args)


@dataclass(frozen=True)
class ComparisonCall(BinaryCall):
    def __bool__(self) -> bool:
        raise NotImplementedError  # TODO: eval to bool


def wrap_function(
    f: Callable[P, R],
    *,
    name: str | None = None,
) -> Callable[P, Call[R]]:
    """Creates a wrapped function that returns a Call expression when called."""
    try:
        name = f.__name__
    except AttributeError:
        if name is None:
            name = "_"

    globals = dict(Call=Call, f=f)

    try:
        parameters = inspect.signature(f).parameters
    except ValueError:
        sig = "(*args, **kwargs)"
    else:
        renamed_parameters = []
        for i, p in enumerate(parameters.values()):
            default = f"{name}_{i}"
            globals[default] = p.default
            renamed_parameters.append(p.replace(default=default))
        sig = str(inspect.Signature(renamed_parameters))

    signature_checker_code = f"def signature_check{sig}: pass"
    create_call_code = "\n\t".join(
        [
            f"def {name}(*args, **kwargs):",
            signature_checker_code,
            "signature_check(*args, **kwargs)",  # TODO: check issue with empty args
            "return Call(f, args, kwargs)",
        ]
    )

    locals = {}
    exec(create_call_code, globals, locals)
    return locals[name]


class Module:
    """Returns wrapped functions from the module on attribute access.

    Wrapped functions return a Call expression when called.
    """

    def __init__(self, module) -> None:
        self._module = module

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._module, name)
        if callable(attr):
            attr = wrap_function(attr, name=name)
        setattr(self, name, attr)
        return attr

    def __repr__(self):
        return f"Module({repr(self._module)}"


def wrap_module(module: R) -> R:
    """Returns wrapped functions from the module on attribute access.

    Wrapped functions return a Call expression when called.
    """
    return Module(module)  # type: ignore
