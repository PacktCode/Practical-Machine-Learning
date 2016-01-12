module TestDimensionalityReductionWrapper

include(joinpath("..", "fixture_learners.jl"))
using .FixtureLearners
fcp = FeatureClassification()

using FactCheck


importall Orchestra.Transformers.DimensionalityReductionWrapper

facts("DimensionalityReduction transformers") do
  context("PCA transforms features") do
    instances = [
      5 10;
      -5 0;
      0 5;
    ]
    labels = ["x"; "y"; "z"]
    options = {:center => false, :scale => false}
    pca = PCA(options)
    fit!(pca, instances, labels)
    transformed = transform!(pca, instances)

    @fact true => maximum(instances - transformed * pca.model.rotation') < 10e-4
  end
end

end # module
