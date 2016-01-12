# RandomForests.jl

CART-based random forest implementation in Julia.
This package supports:
* Classification model
* Regression model
* Out-of-bag (OOB) error
* Feature importances
* Various configurable parameters
**Please be aware that this package is not yet fully examined implementation. You can use it at your own risk.**
And your bug report or suggestion is welcome!
```julia
RandomForestClassifier(;n_estimators::Int=10,
                        max_features::Union(Integer, FloatingPoint, Symbol)=:sqrt,
                        max_depth=nothing,
                        min_samples_split::Int=2,
                        criterion::Symbol=:gini)
```

```julia
RandomForestRegressor(;n_estimators::Int=10,
                       max_features::Union(Integer, FloatingPoint, Symbol)=:third,
                       max_depth=nothing,
                       min_samples_split::Int=2)
```

* `n_estimators`: the number of weak estimators
* `max_features`: the number of candidate features at each split
    * if `Integer` is given, the fixed number of features are used
    * if `FloatingPoint` is given, the proportion of given value (0.0, 1.0] are used
    * if `Symbol` is given, the number of candidate features is decided by a strategy
        * `:sqrt`: `ifloor(sqrt(n_features))`
        * `:third`: `div(n_features, 3)`
* `max_depth`: the maximum depth of each tree
    * the default argument `nothing` means there is no limitation of the maximum depth
* `min_samples_split`: the minimum number of sub-samples to try to split a node
* `criterion`: the criterion of impurity measure (classification only)
    * `:gini`: Gini index
    * `:entropy`: Cross entropy

`RandomForestRegressor` always uses the mean squared error for its impurity measure.
At the current moment, there is no configurable criteria for regression model.


## Related package
* [DecisionTree.jl]
    * DecisionTree.jl is based on the ID3 (Iterative Dichotomiser 3) algorithm while RandomForests.jl uses CART (Classification And Regression Tree).

## Acknowledgement
The algorithm and interface are highly inspired by those of [scikit-learn](http://scikit-learn.org).
