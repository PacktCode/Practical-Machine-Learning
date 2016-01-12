# Practical Machine learning
# k-Nearest Neighbors
# Chapter 6

#typeData could be either "train" or "test.
#labelsInfo should contain the IDs of each image to be read
#The images in the trainResized and testResized data files
#are 20x20 pixels, so imageSize is set to 400.
#path should be set to the location of the data files.

function read_data(typeData, labelsInfo, imageSize, path)
 #Intialize x matrix
 x = zeros(size(labelsInfo, 1), imageSize)

 for (index, idImage) in enumerate(labelsInfo["ID"]) 
  #Read image file 
  nameFile = "$(typeData)Resized/$(idImage).Bmp"
  img = imread(nameFile)

  #Convert img to float values 
  temp = float32sc(img)

  #Convert color images to gray images
  #by taking the average of the color scales. 
  if ndims(temp) == 3
   temp = mean(temp.data, 1)
  end
    
  #Transform image matrix to a vector and store 
  #it in data matrix 
  x[index, :] = reshape(temp, 1, imageSize)
 end 
 return x
end

imageSize = 400 # 20 x 20 pixel

#Read information about training data , IDs.
labelsInfoTrain = readtable("training.csv")

#Read training matrix
xTrain = read_data("train", labelsInfoTrain, imageSize, path)

#Read information about test data ( IDs ).
labelsInfoTest = readtable("sampledata.csv")

#Read test matrix
xTest = read_data("test", labelsInfoTest, imageSize, path)

#Get only first character of string (convert from string to character).
#Apply the function to each element of the column "Class"
yTrain = map(x -> x[1], labelsInfoTrain["Class"])

#Convert from character to integer
yTrain = int(yTrain)

xTrain = xTrain'
xTest = xTest'

addprocs(3) 

@everywhere function euclidean_distance(a, b)
 distance = 0.0 
 for index in 1:size(a, 1) 
  distance += (a[index]-b[index]) * (a[index]-b[index])
 end
 return distance
end

@everywhere function get_k_nearest_neighbors(x, i, k)
 nRows, nCols = size(x)
 imageI = Array(Float32, nRows)
 for index in 1:nRows
  imageI[index] = x[index, i]
 end
 imageJ = Array(Float32, nRows)
 distances = Array(Float32, nCols) 
 for j in 1:nCols
  for index in 1:nRows
   imageJ[index] = x[index, j]
  end
  distances[j] = euclidean_distance(imageI, imageJ)
 end
 sortedNeighbors = sortperm(distances)
 kNearestNeighbors = sortedNeighbors[2:k+1]
 return kNearestNeighbors
end 

@everywhere function assign_label(x, y, k, i)
 kNearestNeighbors = get_k_nearest_neighbors(x, i, k) 
 counts = Dict{Int, Int}() 
 highestCount = 0
 mostPopularLabel = 0
 for n in kNearestNeighbors
  labelOfN = y[n]
  if !haskey(counts, labelOfN)
   counts[labelOfN] = 0
  end
  counts[labelOfN] += 1 
  if counts[labelOfN] > highestCount
   highestCount = counts[labelOfN]
   mostPopularLabel = labelOfN
  end 
 end
 return mostPopularLabel
end

tic()
k = 1
sumValues = @parallel (+) for i in 1:size(xTrain, 2)
 assign_label(xTrain, yTrain, k, i) == yTrain[i, 1]
end
loofCvAccuracy = sumValues / size(xTrain, 2)
println("The LOOF-CV accuracy of 1NN is $(loofCvAccuracy)")
toc()

#Similar to function assign_label.
#Only changes are commented
@everywhere function assign_label_each_k(x, y, maxK, i)
 kNearestNeighbors = get_k_nearest_neighbors(x, i, maxK) 

 #The next array will keep the labels for each value of k
 labelsK = zeros(Int, 1, maxK) 

 counts = Dict{Int, Int}()
 highestCount = 0
 mostPopularLabel = 0

 #We need to keep track of the current value of k
 for (k, n) in enumerate(kNearestNeighbors)
  labelOfN = y[n]
  if !haskey(counts, labelOfN)
   counts[labelOfN] = 0
  end
  counts[labelOfN] += 1
  if counts[labelOfN] > highestCount
   highestCount = counts[labelOfN]
   mostPopularLabel = labelOfN  
  end
  #Save current most popular label 
  labelsK[k] = mostPopularLabel
 end
 #Return vector of labels for each k
 return labelsK
end

tic()
maxK = 20 #Any value can be chosen
yPredictionsK = @parallel (vcat) for i in 1:size(xTrain, 2)
 assign_label_each_k(xTrain, yTrain, maxK, i)
end
for k in 1:maxK
 accuracyK = mean(yTrain .== yPredictionsK[:, k])
 println("The LOOF-CV accuracy of $(k)-NN is $(accuracyK)")
end
toc()

@everywhere function get_k_nearest_neighbors(xTrain, imageI, k)
 nRows, nCols = size(xTrain) 
 imageJ = Array(Float32, nRows)
 distances = Array(Float32, nCols) 
 for j in 1:nCols
  for index in 1:nRows
   imageJ[index] = xTrain[index, j]
  end
  distances[j] = euclidean_distance(imageI, imageJ)
 end
 sortedNeighbors = sortperm(distances)
 kNearestNeighbors = sortedNeighbors[1:k]
 return kNearestNeighbors
end 

@everywhere function assign_label(xTrain, yTrain, k, imageI)
 kNearestNeighbors = get_k_nearest_neighbors(xTrain, imageI, k) 
 counts = Dict{Int, Int}() 
 highestCount = 0
 mostPopularLabel = 0
 for n in kNearestNeighbors
  labelOfN = yTrain[n]
  if !haskey(counts, labelOfN)
   counts[labelOfN] = 0
  end
  counts[labelOfN] += 1 #add one to the count
  if counts[labelOfN] > highestCount
   highestCount = counts[labelOfN]
   mostPopularLabel = labelOfN
  end 
 end
 return mostPopularLabel
end

println("Running kNN on test data")
tic()
k = 3 # The CV accuracy shows this value to be the best.
yPredictions = @parallel (vcat) for i in 1:size(xTest, 2)
 nRows = size(xTrain, 1)
 imageI = Array(Float32, nRows)
 for index in 1:nRows
  imageI[index] = xTest[index, i]
 end
 assign_label(xTrain, yTrain, k, imageI)
end
toc()

#Convert integer predictions to character
labelsInfoTest["Class"] = char(yPredictions)

#Save predictions
writetable("$(path)/juliaKNNSubmission.csv", labelsInfoTest, separator=',', header=true)

println("Submission file saved in $(path)/juliaKNNSubmission.csv")
Status API Training Shop Blog About Pricing
© 2016 GitHub, Inc. Terms Privacy Security Contact Help