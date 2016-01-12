# Practical Machine learning
# Ensemble learning example 
# Chapter 13

# Dimensionality Reduction transformers.
module DimensionalityReductionWrapper

importall Orchestra.Types
importall Orchestra.Util
import DimensionalityReduction: pca

export PCA,
       fit!,
       transform!

# Principal Component Analysis rotation
# on features.
# Features ordered by maximal variance descending.
#
# Fails if zero-variance feature exists.
type PCA <: Transformer
  model
  options

  function PCA(options=Dict())
    default_options = {
      :center => true,
      :scale => true
    }
    new(nothing, nested_dict_merge(default_options, options))
  end
end

function fit!(p::PCA, instances::Matrix, labels::Vector)
  pca_model = pca(instances; p.options...)
  p.model = pca_model
end

function transform!(p::PCA, instances::Matrix)
  return instances * p.model.rotation
end

end # module
