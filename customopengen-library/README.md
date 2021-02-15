## OpenGen

Code generation tool for OpEn with minor edits compared to original GitHub project, allowing for parametric rectangle bounds and better TCP sockets.

### Installation 
To use this custom Python package, run:

```
python setup.py bdist_wheel
pip install <path/to/generated/whl>
```

You will then be able to use it in your project with

```python
from customopengen import *
```

or 

```python
import customopengen as og
```

In order to build the auto-generated code, you need the 
Rust compiler.

For detailed documentation, please refer to 
[OpEn's website](https://alphaville.github.io/optimization-engine/).

## Citing OpEn

Please, cite OpEn as follows:

```
@inproceedings{open2020,
author="P. Sopasakis and E. Fresk and P. Patrinos",
title="OpEn: Code Generation for Embedded Nonconvex Optimization",
booktitle="IFAC World Congress",
year="2020",
address="Berlin"
}
```
 