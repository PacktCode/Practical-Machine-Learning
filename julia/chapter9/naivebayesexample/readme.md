Naive Bayes classifier. Currently 2 types of NB are supported: 

 * **MultinomialNB** - assumes variables have a multinomial distribution. Good e.g. for text classification. See `examples/nums.jl` for usage.
 * **GaussianNB** - assumes variables have a multivariate normal distribution. Good for real-valued data. See `examples/iris.jl` for usage.

Since `GaussianNB` models multivariate distribution, it's not really a "naive" classifier (i.e. no independence assumption is made), so the name may change in the future. 
