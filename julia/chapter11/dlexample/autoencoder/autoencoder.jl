# Practical Machine learning
# Deep learning - Sparse autoencoder example 
# Chapter 11

module SparseAutoencoder

function sigmoid(z)
        return one(z[1]) / (one(z[1]) + exp(-z))
end

function autoencodercost!(theta::Array{Float64,1}, visiblesize, hiddensize, data::Array{Float64,2}, grad::Vector{Float64};
                          lambda=0.0, sparsityparameter=0.01, beta=0.0)
        W1 = reshape(theta[1:hiddensize*visiblesize],hiddensize,visiblesize)
        W2 = reshape(theta[hiddensize*visiblesize+1:2*hiddensize*visiblesize],visiblesize,hiddensize)
        b1 = theta[2*hiddensize*visiblesize+1:2*hiddensize*visiblesize+hiddensize]
        b2 = theta[2*hiddensize*visiblesize+hiddensize+1:end]
        m = size(data,2)

        # Forward pass
        a1 = data
        z2 = broadcast(+,W1*a1,b1)
        a2 = sigmoid(z2)
        z3 = broadcast(+,W2*a2,b2)
        a3 = sigmoid(z3)
        output = a3

        avghiddenactivity = mean(a2,2)
        sparsitycost = sum(sparsityparameter*log(sparsityparameter/avghiddenactivity) + (1 - sparsityparameter)*log((1-sparsityparameter)/(1-avghiddenactivity)))
        cost = sum((output - data).^2)/(2*m) + lambda/2*sum(theta[1:2*hiddensize*visiblesize].^2) + beta*sparsitycost

        if length(grad) > 0
            # Back-propagation for sparsified objective
            a3prime = a3.*(one(a3[1])-a3)
            a2prime = a2.*(one(a2[1])-a2)
            sparsityterm = beta*(-sparsityparameter/avghiddenactivity + (1-sparsityparameter)/(1-avghiddenactivity))

            d3 = -(data - a3).* a3prime
            d2 = broadcast(+,W2'*d3,sparsityterm).*a2prime

            grad[1:visiblesize*hiddensize] = (d2*a1'/m + lambda*W1)[:] #dW1
            grad[visiblesize*hiddensize+1:2*visiblesize*hiddensize] = (d3*a2'/m + lambda*W2)[:] #dW2
            grad[2*visiblesize*hiddensize+1:2*hiddensize*visiblesize+hiddensize] = (sum(d2,2)/m)[:]
            grad[2*hiddensize*visiblesize+hiddensize+1:end] = (sum(d3,2)/m)[:]
        end
        return cost

end

function numericalgradient(f,x::Array{Float64,1})
        epsilon = 1e-4
        g = zeros(length(x))
        for i =1:length(x)
                x[i] += epsilon
                fplus = f(x)
                x[i] -= 2*epsilon
                fminus = f(x)
                g[i] = (fplus - fminus)/(2*epsilon)
                x[i] += epsilon
         end
         return g
end

function testGradient()
        vars = matread("IMAGES.mat")
        traindata = Array(Float64,SAMPLE_WIDTH*SAMPLE_HEIGHT,100)
        sampleimages!(vars["IMAGES"],traindata,100)

        hidden = 10
        visible = 64
        lambda = 1e-4
        beta = 3.0
        sparsityparameter=0.01
        theta = 0.1*randn(2*hidden*visible+hidden+visible)
        function justcost(x)
                c,g = autoencodercost(x,visible,hidden,traindata,lambda=lambda,sparsityparameter=sparsityparameter,beta=beta)
                return c
        end
        cost,grad = autoencodercost(theta,visible,hidden,traindata,lambda=lambda,sparsityparameter=sparsityparameter,beta=beta)
        numgrad = numericalgradient(justcost,theta)
        println([grad numgrad])
        diff = norm(numgrad - grad) / norm(numgrad + grad)^2
        return diff
end

function initializeparameters(hiddensize,visiblesize)
        r = sqrt(6) / sqrt(hiddensize+visiblesize+1)
        W1 = rand(hiddensize,visiblesize)*2*r - r
        W2 = rand(visiblesize,hiddensize)*2*r - r
        b1 = zeros(hiddensize)
        b2 = zeros(visiblesize)
        return [W1[:],W2[:],b1,b2]
end

using Plotly
export displaynetwork
function displaynetwork(A,filename)
        m,n = size(A)
        sz = int(sqrt(m))
        A -= mean(A)
        layout = [
            "autosize" => false,
            "width" => 500,
            "height"=> 500
        ]

        gridsize = int(ceil(sqrt(n)))
        buffer = 1
        griddata = ones(gridsize*(sz+1)+1,gridsize*(sz+1)+1)
        index = 1
        for i = 1:gridsize
                for j = 1:gridsize
                        if index > n
                                continue
                        end
                        columnlimit = maximum(abs(A[:,index]))
                        griddata[buffer+(i-1)*(sz+buffer)+(1:sz),buffer+(j-1)*(sz+buffer)+(1:sz)] = reshape(A[:,index],sz,sz)/columnlimit
                        index += 1
                end
        end

        Plotly.signin("kjchavez", "dd16t6j7li")
        data = [
          [
            "z" => griddata,
            "colorscale" => "Greys",
            "type" => "heatmap"
          ]
        ]
        response = Plotly.plot(data, ["layout" => layout, "filename" => filename, "fileopt" => "overwrite"])
        plot_url = response["url"]
end

using NLopt
export autoencode
function autoencode(data,hiddensize,visiblesize;lambda=1e-4,beta=3.0,rho=0.01,maxiter=600)
        theta::Vector{Float64} = initializeparameters(hiddensize,visiblesize)
        count = 0
        function objective(x,grad)
                count += 1
                println("iteration $count")
                return autoencodercost!(x,visiblesize,hiddensize,data, grad, lambda=lambda,sparsityparameter=rho,beta=beta)
        end
        opt = Opt(:LD_LBFGS,length(theta))
        min_objective!(opt,objective)
        maxeval!(opt,maxiter)
        minf,minx,ret = optimize(opt,theta)
        W1 = reshape(minx[1:hiddensize*visiblesize],hiddensize,visiblesize)
        W2 = reshape(minx[hiddensize*visiblesize+1:2*hiddensize*visiblesize],visiblesize,hiddensize)
        b1 = minx[2*hiddensize*visiblesize+1:2*hiddensize*visiblesize+hiddensize]
        b2 = minx[2*hiddensize*visiblesize+hiddensize+1:end]

        return minf,W1,W2,b1,b2
end

using MAT
const SAMPLE_WIDTH = 8
const SAMPLE_HEIGHT = 8
function sampleimages(images::Array{Float64,3},numsamples)
    width, height = size(images[:,:,1])
    array::Array{Float64,2} = zeros(SAMPLE_WIDTH*SAMPLE_HEIGHT,numsamples)
    for index=1:numsamples
        image_index = rand(1:size(images,3))
        x = rand(1:width-SAMPLE_WIDTH+1)
        y = rand(1:height-SAMPLE_HEIGHT+1)
        sample = images[x:x+SAMPLE_WIDTH-1,y:y+SAMPLE_HEIGHT-1,image_index]
        array[:,index] = reshape(sample,SAMPLE_WIDTH*SAMPLE_HEIGHT)
        array[:,index] -= mean(array[:,index]) #subtract mean
    end

    # rescale images to fit in range 0.1 to 0.9
    stddev = std(array)
    array = max(min(array,3*stddev),-3*stddev) / (3*stddev)
    array = (array + 1.0) * 0.4 + 0.1
    return array
end

function main()
        vars = matread("IMAGES.mat")
        traindata = sampleimages(vars["IMAGES"],10000)

        hiddensize = 25
        visiblesize = 64

        # tunable parameters
        lambda = 1e-4
        beta = 3.0
        rho = 0.01

        f,W1,W2,b1,b2 = autoencode(traindata,hiddensize,visiblesize)
        displaynetwork(W1',"autoencoder-tutorial")
end

end
