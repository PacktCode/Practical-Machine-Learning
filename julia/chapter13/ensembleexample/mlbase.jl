# Practical Machine learning
# Ensemble learning example 
# Chapter 13

# MLBase transformers.
module MLBaseWrapper

importall Orchestra.Types
importall Orchestra.Util

import MLBase: Standardize, estimate, transform

export StandardScaler,
       fit!,
       transform!

# Standardizes each feature using (X - mean) / stddev.
# Will produce NaN if standard deviation is zero.
type StandardScaler <: Transformer
  model
  options

  function StandardScaler(options=Dict())
    default_options = {
      :center => true,
      :scale => true
    }
    new(nothing, nested_dict_merge(default_options, options))
  end
end

function fit!(st::StandardScaler, instances::Matrix, labels::Vector)
  st_transform = estimate(Standardize, instances'; st.options...)
  st.model = {
    :standardize_transform => st_transform
  }
end

function transform!(st::StandardScaler, instances::Matrix)
  st_transform = st.model[:standardize_transform]
  transposed_instances = instances'
  return transform(st_transform, transposed_instances)'
end

end # module
