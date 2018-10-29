#helpers.jl
module helpers
import ImageMagick
using Compat, GZip, SparseArrays, Images, Statistics, Knet

function onehot(a, atype)
	yrows = 10
	a[a .== 0] .= 10
	sparseVec = sparse(a, 1:length(a), ones(eltype(a), length(a)), yrows, length(a))
	return convert(atype, sparseVec)
end

function gzload(file; path="$file", url="http://yann.lecun.com/exdb/mnist/$file")
	isfile(path) || download(url, path)
	f = gzopen(path)
	a = @compat read(f)
	close(f)
	return(a)
end

function loaddata(atype; dataset="MNIST", N=nothing)
	if dataset == "MNIST"
		xDim, yDim = 28, 28
	    # @info "Loading MNIST..."
	    xtrn = gzload("train-images-idx3-ubyte.gz")[17:end]
	    xtst = gzload("t10k-images-idx3-ubyte.gz")[17:end]
	    ytrn = gzload("train-labels-idx1-ubyte.gz")[9:end]
	    ytst = gzload("t10k-labels-idx1-ubyte.gz")[9:end]

		# Preprocessing and reshaping
	    xtrn, xtst = map(x-> (x./(255/2)) .-1, (xtrn, xtst))
	    xtrn, xtst = map(x-> reshape(x, xDim, yDim, 1, :), (xtrn, xtst))
	    xtrn, xtst = map(x-> convert(atype, x), (xtrn, xtst))

	    # One-Hot encoding
	    ytrn, ytst = map(a->onehot(a, atype), (ytrn, ytst))
	else
		# N=15448
		xDim, yDim = 64, 64
		x = atype(undef, (xDim, yDim, 1, N))

		count = 1
		for imgfile = readdir(dataset)
			count>N && break
			fullfile = string(dataset, "/", imgfile)
			img = convert(atype, channelview(Gray.(load(fullfile))))
			x[:,:,1,count] = img
			count+=1
		end

		x .-= mean(x; dims=(1,2,3))
		x ./= std(x; dims=(1,2,3))
		trnSize = Int(round(0.9 * N))
		xtrn = x[:,:,:, 1:trnSize]
		xtst = x[:,:,:, trnSize+1:end]
		ytrn = []; ytst = [];
	end
    return (xtrn, xtst, ytrn, ytst)
end
export loaddata


function mbatch(idx, batchsize)
    data = Any[]
    for i = 1:batchsize:(size(idx)[end]-batchsize+1)
		j = i + batchsize - 1
		push!(data, idx[i:j])
    end
    return data
end
export mbatch


function get_params(params, atype)
    len = length(params["value"])
    ws, ms = [], []
    for k = 1:len
        name = params["name"][k]
        value = convert(Array{Float32}, params["value"][k])

        if endswith(name, "moments")
            push!(ms, reshape(value[:,1], (1,1,size(value,1),1)))
            push!(ms, reshape(value[:,2], (1,1,size(value,1),1)))
        elseif startswith(name, "bn")
            push!(ws, reshape(value, (1,1,length(value),1)))
        elseif startswith(name, "fc") && endswith(name, "filter")
            push!(ws, transpose(reshape(value,(size(value,3),size(value,4)))))
        elseif startswith(name, "conv") && endswith(name, "bias")
            push!(ws, reshape(value, (1,1,length(value),1)))
        else
            push!(ws, value)
        end
    end
    map(wi->convert(atype, wi), ws),
    map(mi->convert(atype, mi), ms)
end
export get_params


function makegrid(y; gridsize=[10,10], scale=2.0, shape=(64,64))
    y = map(i->y[:, :, 1, i]', [1:size(y,4)...])
    shp = map(x->Int(round(x*scale)), shape)
    y = map(x->Images.imresize(x,shp), y)
    gridx, gridy = gridsize
    outdims = (gridx*shp[1] + gridx + 1, (gridy*shp[2]) + gridy + 1)
    out = zeros(outdims...)
    for k = 1:gridx+1; out[(k-1)*(shp[1]+1)+1,:] .= 1.0; end
    for k = 1:gridy+1; out[:,(k-1)*(shp[2]+1)+1] .= 1.0; end

    x0 = y0 = 2
    for k = 1:length(y)
        x1 = x0+shp[1]-1
        y1 = y0+shp[2]-1
        out[x0:x1,y0:y1] = y[k]

        y0 = y1+2
        if k % gridy == 0
			x0 = x1+2
			y0 = 2
		else
			y0 = y1+2
        end
    end

    return convert(Array{Float64,2}, map(x->isnan(x) ? 0 : x, out))
end
export makegrid

function save_weights(model; savefile=nothing)
	if savefile==nothing
		save("F.jld2", Dict("F" => map(x->Array(x), model.F.w)))
		save("D.jld2", Dict("D"=> map(x->Array(x), model.D.w)))
		save("G.jld2", Dict("G" => map(x->Array(x), model.G.w)))
		save("Q.jld2", Dict("Q"=> map(x->Array(x), model.Q.w)))
	else
		save(savefile,
			Dict(
			"F" => map(x->Array(x), model.F.w),
			"D" => map(x->Array(x), model.D.w),
			"G" => map(x->Array(x), model.G.w),
			"Q" => map(x->Array(x), model.Q.w),
			))
	end
end
export save_weights

function resnet50(w, x; mode=true)
    conv1  = conv4(w[1],x; padding=3, stride=2) .+ w[2]
    bn1    = batchnorm(conv1, bnmoments(); training=mode)
    pool1  = pool(bn1; window=3, stride=2)

    # layer 2,3,4,5
    r2 = reslayerx5(w[5:34], pool1; strides=[1,1,1,1], mode=mode)
    r3 = reslayerx5(w[35:73], r2; mode=mode)
    r4 = reslayerx5(w[74:130], r3; mode=mode) # 5
    r5 = reslayerx5(w[131:160], r4; mode=mode)

    # fully connected layer
    pool5  = pool(r5; stride=1, window=2, mode=2)
    fc1000 = w[161] * mat(pool5) .+ w[162]
end
export resnet50


function reslayerx0(w, x; padding=0, stride=1, mode=true)
    b  = conv4(w[1], x; padding=padding, stride=stride)
    bx = batchnorm(b, bnmoments(); training=mode)
end


function reslayerx1(w, x; padding=0, stride=1, mode=0)
    relu.(reslayerx0(w, x; padding=padding, stride=stride, mode=mode))
end


function reslayerx2(w,x; pads=[0,1,0], strides=[1,1,1], mode=0)
    ba = reslayerx1(w[1:3],x; padding=pads[1], stride=strides[1], mode=mode)
    bb = reslayerx1(w[4:6],ba; padding=pads[2], stride=strides[2], mode=mode)
    bc = reslayerx0(w[7:9],bb; padding=pads[3], stride=strides[3], mode=mode)
end


function reslayerx3(w,x; pads=[0,0,1,0], strides=[2,2,1,1], mode=0) # 12
    a = reslayerx0(w[1:3],x; stride=strides[1], padding=pads[1], mode=mode)
    b = reslayerx2(w[4:12],x; strides=strides[2:4], pads=pads[2:4], mode=mode)
    relu.(a .+ b)
end


function reslayerx4(w,x; pads=[0,1,0], strides=[1,1,1], mode=0)
    relu.(x .+ reslayerx2(w,x; pads=pads, strides=strides, mode=mode))
end


function reslayerx5(w,x; strides=[2,2,1,1], mode=0)
    x = reslayerx3(w[1:12],x; strides=strides, mode=mode)
    for k = 13:9:length(w)
        x = reslayerx4(w[k:k+8],x; mode=mode)
    end
    return x
end

end
