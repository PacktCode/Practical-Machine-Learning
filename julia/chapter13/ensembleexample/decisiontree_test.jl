module TestDecisionTreeWrapper

include(joinpath("..", "fixture_learners.jl"))
using .FixtureLearners
nfcp = NumericFeatureClassification()

using FactCheck


importall Orchestra.Transformers.DecisionTreeWrapper
using DecisionTree

facts("DecisionTree learners") do
  context("PrunedTree gives same results as its backend") do
    # Predict with Orchestra learner
    learner = PrunedTree()
    orchestra_predictions = fit_and_transform!(learner, nfcp)

    # Predict with original backend learner
    srand(1)
    model = build_tree(nfcp.train_labels, nfcp.train_instances)
    model = prune_tree(model, 1.0)
    original_predictions = apply_tree(model, nfcp.test_instances)

    # Verify same predictions
    @fact orchestra_predictions => original_predictions
  end

  context("RandomForest gives same results as its backend") do
    # Predict with Orchestra learner
    learner = RandomForest()
    orchestra_predictions = fit_and_transform!(learner, nfcp)

    # Predict with original backend learner
    srand(1)
    model = build_forest(
      nfcp.train_labels,
      nfcp.train_instances,
      size(nfcp.train_instances, 2),
      10,
      0.7
    )
    original_predictions = apply_forest(model, nfcp.test_instances)

    # Verify same predictions
    @fact orchestra_predictions => original_predictions
  end

  context("DecisionStumpAdaboost gives same results as its backend") do
    # Predict with Orchestra learner
    learner = DecisionStumpAdaboost()
    orchestra_predictions = fit_and_transform!(learner, nfcp)

    # Predict with original backend learner
    srand(1)
    model, coeffs = build_adaboost_stumps(
      nfcp.train_labels,
      nfcp.train_instances,
      7
    )
    original_predictions = apply_adaboost_stumps(
      model, coeffs, nfcp.test_instances
    )

    # Verify same predictions
    @fact orchestra_predictions => original_predictions
  end

  context("RandomForest handles training-dependent options") do
    # Predict with Orchestra learner
    learner = RandomForest({:impl_options => {:num_subfeatures => 2}})
    orchestra_predictions = fit_and_transform!(learner, nfcp)

    # Verify RandomForest didn't die
    @fact 1 => 1
  end
end

end # module
