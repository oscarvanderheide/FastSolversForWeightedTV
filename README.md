# FastSolversForWeightedTV

Set of utilities for computing proximal and projection operators with the TV regularization, and weighted version thereof. The methods are applicable to 2D and 3D problems and support GPU.

To install:
```
] add https://github.com/grizzuti/FastSolversForWeightedTV.git
```

See examples in the folder /examples for applications of total variation regularization via proxy or projection.

This package is highly indebted to [ProximalOperators.jl](https://github.com/JuliaFirstOrder/ProximalOperators.jl) and related projects.

## References

 - Matthias J. Ehrhardt, Marta M. Betcke, "Multi-Contrast MRI Reconstruction with Structure-Guided Total Variation", SIAM Journal on Imaging Sciences, 2016. https://arxiv.org/abs/1511.06631