from dataclasses import dataclass

import numpy as np

import usymbol


@dataclass(frozen=True)
class MySymbol(usymbol.Symbol):
    name: str

    def __str__(self):
        return self.name


np = usymbol.wrap_module(np)

x = MySymbol("x")
cos = MySymbol("cos")

y = np.cos(x) + cos
z = usymbol.substitute(
    y,
    {
        np._module.cos: np._module.sin,  # falta hacer esto bien
        x: x + np.pi / 2,
    },
)

print(y)
print(z)

print(f"{usymbol.substitute_and_evaluate(y, {x: 1, cos: 0})=}")
print(f"{usymbol.substitute_and_evaluate(z, {x: 1, cos: 0})=}")

f_y = usymbol.to_function(y, [x, cos], name="y")
print(f"{f_y(1, 0)=}")
print(f"{f_y(x=1, cos=0)=}")
print(f"{f_y(cos=0, x=1)=}")
print(f"{f_y(cos=1, x=0)=}")
