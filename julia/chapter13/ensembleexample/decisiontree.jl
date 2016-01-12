# Practical Machine learning
# Ensemble learning example 
# Chapter 13

# Decision trees as found in DecisionTree Julia package.
module DecisionTreeWrapper

importall Orchestra.Types
importall Orchestra.Util

import DecisionTree
DT = DecisionTree

export PrunedTree, 
       RandomForest,
       DecisionStumpAdaboost,
       fit!, 
       transform!

# Pruned ID3 decision tree.
type PrunedTree <: Learner
  model
  options
  
  function PrunedTree(options=Dict())
    default_options = {
      # Output to train against
      # (:class).
      :output => :class,
      # Options specific to this implementation.
      :impl_options => {
        # Merge leaves having >= purity_threshold combined purity.
        :purity_threshold => 1.0
      }
    }
    new(nothing, nested_dict_merge(default_options, options))
  end
end

function fit!(tree::PrunedTree, instances::Matrix, labels::Vector)
  impl_options = tree.options[:impl_options]
  tree.model = DT.build_tree(labels, instances)
  tree.model = DT.prune_tree(tree.model, impl_options[:purity_threshold])
end
function transform!(tree::PrunedTree, instances::Matrix)
  return DT.apply_tree(tree.model, instances)
end

# Random forest (C4.5).
type RandomForest <: Learner
  model
  options
  
  function RandomForest(options=Dict())
    default_options = {
      # Output to train against
      # (:class).
      :output => :class,
      # Options specific to this implementation.
      :impl_options => {
        # Number of features to train on with trees.
        :num_subfeatures => nothing,
        # Number of trees in forest.
        :num_trees => 10,
        # Proportion of trainingset to be used for trees.
        :partial_sampling => 0.7
      }
    }
    new(nothing, nested_dict_merge(default_options, options))
  end
end

function fit!(forest::RandomForest, instances::Matrix, labels::Vector)
  # Set training-dependent options
  impl_options = forest.options[:impl_options]
  if impl_options[:num_subfeatures] == nothing
    num_subfeatures = size(instances, 2)
  else
    num_subfeatures = impl_options[:num_subfeatures]
  end
  # Build model
  forest.model = DT.build_forest(
    labels, 
    instances,
    num_subfeatures, 
    impl_options[:num_trees],
    impl_options[:partial_sampling]
  )
end

function transform!(forest::RandomForest, instances::Matrix)
  return DT.apply_forest(forest.model, instances)
end

# Adaboosted C4.5 decision stumps.
type DecisionStumpAdaboost <: Learner
  model
  options
  
  function DecisionStumpAdaboost(options=Dict())
    default_options = {
      # Output to train against
      # (:class).
      :output => :class,
      # Options specific to this implementation.
      :impl_options => {
        # Number of boosting iterations.
        :num_iterations => 7
      }
    }
    new(nothing, nested_dict_merge(default_options, options))
  end
end

function fit!(adaboost::DecisionStumpAdaboost, 
  instances::Matrix, labels::Vector)

  # NOTE(svs14): Variable 'model' renamed to 'ensemble'.
  #              This differs to DecisionTree
  #              official documentation to avoid confusion in variable
  #              naming within Orchestra.
  ensemble, coefficients = DT.build_adaboost_stumps(
    labels, instances, adaboost.options[:impl_options][:num_iterations]
  )
  adaboost.model = {
    :ensemble => ensemble,
    :coefficients => coefficients
  }
end

function transform!(adaboost::DecisionStumpAdaboost, instances::Matrix)
  return DT.apply_adaboost_stumps(
    adaboost.model[:ensemble], adaboost.model[:coefficients], instances
  )
end

end # module
