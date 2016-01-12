module PlottingUtils

function reshapedata(A)
        m,n = size(A)
        sz = int(sqrt(m))
        A -= mean(A)

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

	return griddata
end

import Plotly
export displaynetwork_plotly
function displaynetwork_plotly(A,filename,username,userkey)
	griddata = reshapedata(A)
	Plotly.signin(username, userkey)
        data = [
          [
            "z" => griddata,
            "colorscale" => "Greys",
            "type" => "heatmap"
          ]
        ]
        layout = [
            "autosize" => false,
            "width" => 500,
            "height"=> 500
        ]
        response = Plotly.plot(data, ["layout" => layout, "filename" => filename, "fileopt" => "overwrite"])
        plot_url = response["url"]
end

import Gadfly
export displaynetwork_gadfly
function displaynetwork_gadfly(A)
	griddata = reshapedata(A)
	Gadfly.spy(A)
end

import Winston
export displaynetwork_winston
function displaynetwork_winston(A)
	griddata = reshapedata(A)
	p = Winston.FramedPlot()
	Winston.colormap("grays")
	Winston.add(p,Winston.imagesc(griddata))
	display(p)
end

end
