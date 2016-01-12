# Practical Machine learning
# Clustering based learning - L-mena clustering 
# Chapter 8

using Images
using ImageView
using Color

# Run non-interactively, accept either an image or a directory
# If directory, iterate through image filetypes

# Enter filepath and number of dominant colors wanted, k
function dominant_colors (filename, k) 
	if isfile(filename)
		img = imread(filename,RGB)
		init(img, filename, k)
	elseif isdir(filename)
		files = filter!(abspath(r"\.(?:jpe?g|gif|png|tiff)$")), readdir()) #creates an array of filenames by filtering out only files that in file extensions.
		for i in files
			dominant_colors(i)
		end
	else 
		error("No image found.")
	end
end

type Point
	coords::Array # Color associated with pt, a 3D array.
	ct:Int # Count
end

type Cluster
	points::Array # Points associated with cluster, C_k
	centroid::Point # Center of cluster, assumed mean of pt values
	k::Int # Cluster count
end

function init(img, filename, k)
# Convert color space from sRGB (linear) to CIEXYZ to CIELAB
	run(`convert $filename -thumbnail 200x200 $filename`) #convert to thumbnail via ImageMagick CLI
	img = convert(Image{LAB}, img) #use Color to convert to LAB automagically
	points = getpoints(img)
	randclusters(points, k)
	kmeans(points, k) 
	
function getpoints(img)
	points = []
	count = 0
	for count, color in img[1:width(img)]
		for count, color in img[2:height(img)]
			count += 1
			points.append(Point(color, count))
		end
	return points
end	

# Sq. euclidean distance
function distance (pt1, pt2)
	return mapreduce((pt1.coords[i]-pt2.coords[i])**2, +, 1:length(pt1.coords))
end

# Randomly assign pixels to represent intial centroid/clusters
function randclusters(pts::Array, k)
	kclusters = []
	for n = 1:k 
		kclusters.append(Cluster(pts, pts[rand(1:end)], n)
	return kclusters
end

# Recalculate Centroid
function recenter(points, )

# K-Means Algorithm (alternate between resigning points to a cluster based on similarity and cluster centroid based on the points assigned)
function kmeans()
# Repeat n number of times 

# Optionally convert returned clusters back to sRGB