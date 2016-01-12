module DataUtils

export sampleimages
function sampleimages(images::Array{Float64,3},patchwidth::Int,patchheight::Int,numsamples::Int; scalevariance=true)
    width, height = size(images[:,:,1])
    array::Array{Float64,2} = zeros(patchwidth*patchheight,numsamples)
    for index=1:numsamples
        image_index = rand(1:size(images,3))
        x = rand(1:width-patchwidth+1)
        y = rand(1:height-patchheight+1)
        sample = images[x:x+patchwidth-1,y:y+patchheight-1,image_index]
        array[:,index] = reshape(sample,patchwidth*patchheight)
        array[:,index] -= mean(array[:,index]) #subtract mean
    end

    if scalevariance
        # rescale images to fit in range 0.1 to 0.9
        stddev = std(array)
        array = max(min(array,3*stddev),-3*stddev) / (3*stddev)
        array = (array + 1.0) * 0.4 + 0.1
    end
    return array
end

import Plotly
export displaynetwork_plotly
function displaynetwork_plotly(A,filename,username,userkey)
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

        Plotly.signin(username, userkey)
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

end
