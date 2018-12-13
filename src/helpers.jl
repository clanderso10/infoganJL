#helpers.jl
module helpers
import ImageMagick
using Compat, GZip, Images, Statistics, Random, Printf, Knet
using MAT:matopen


function gzload(file; path="$file", url="http://yann.lecun.com/exdb/mnist/$file")
	isfile(path) || download(url, path)
	f = gzopen(path)
	a = @compat read(f)
	close(f)
	return(a)
end
export gzload

"""
    loaddata(o; keep_p=1.0)

Default dataloader for MNIST dataset or digital holography dataset.


<b> Arguments </b>
- `o::Dict`: Parameter dictionary.
- `keep_p::Float64`: Percentage of y-labels to keep (to tune extent of semisupervision).
- `dataset::String`:  "MNIST" or directory of holography datset .
"""
function loaddata(o::Dict; drop_p=0.0)
	dataset=o[:datadir]
	if dataset == "MNIST"
		xDim, yDim = 28, 28
	    # @info "Loading MNIST..."
	    xtrn = gzload("../resources/train-images-idx3-ubyte.gz")[17:end]
	    xtst = gzload("../resources/t10k-images-idx3-ubyte.gz")[17:end]
	    ytrn = gzload("../resources/train-labels-idx1-ubyte.gz")[9:end]
	    ytst = gzload("../resources/t10k-labels-idx1-ubyte.gz")[9:end]

		# Preprocessing and reshaping
	    xtrn, xtst = map(x-> (x./(255/2)) .-1, (xtrn, xtst))
	    xtrn, xtst = map(x-> reshape(x, xDim, yDim, 1, :), (xtrn, xtst))

	    # One-Hot encoding
	    ytrn, ytst = map(a->reshape(a, 1, :), (ytrn, ytst))
	    ytrn = Int64.(ytrn); ytst = Int64.(ytst)
	    ytrn .+= 1; ytst .+= 1;
	    ytrn, ytst = map(a-> convert(Array{Any}, a), (ytrn, ytst))

	    trnN = size(xtrn,4)

	    #Set percentage of y-data collected
	    idx_hide = randsubseq(1:trnN, drop_p)
	    ytrn[:,idx_hide] .= nothing
	else
		if isfile(string(dataset, "/data.jld2"))
			@info @sprintf("Datafile found. loading %s", string(dataset, "/data.jld2"))
			data = load(string(dataset, "/data.jld2"))
			xtrn = data["xtrn"]
			xtst = data["xtst"]
			ytrn = data["ytrn"]
			ytst = data["ytst"]
		else
			xxDim, xyDim = 64, 64
			yyDim = length(o[:c_SS]);

			x = Array{Float32}(undef, (xxDim, xyDim, 2, N));
			y = Array{Any}(nothing, (yyDim, N));

			pStr, nStr = "_pos_.png", "_neg_.png"
			for cellNum = 1:N
				cellStr = lpad(cellNum, 5, "0")
				f625 = string(dataset, "/", cellStr, "_625")
				f470 = string(dataset, "/", cellStr, "_470")
				isfile(string(f625, pStr)) ? f625=string(f625, pStr) : f625=string(f625, nStr)
				isfile(string(f470, pStr)) ? f470=string(f470, pStr) : f470=string(f470, nStr)

				img625 = convert(Array{Float32}, channelview(Gray.(load(f625))))
				img470 = convert(Array{Float32}, channelview(Gray.(load(f470))))
				x[:,:,1,cellNum] = img625 .- mean(img625; dims=(1,2))
				x[:,:,2,cellNum] = img470 .- mean(img470; dims=(1,2))

				occursin("neg", f625) ? y[1, cellNum] = 1 : y[1,cellNum] = 2
				occursin("neg", f470) ? y[2, cellNum] = 1 : y[2,cellNum] = 2
			end

			trnSize = Int(round(0.9 * N))
			xtrn = x[:,:,:, 1:trnSize]
			xtst = x[:,:,:, trnSize+1:end]
			ytrn = y[:, 1:trnSize]; ytst = y[:, trnSize+1:end];

			save(string(dataset, "/data.jld2"),
					Dict("xtrn" => xtrn,
						"xtst" => xtst,
						"ytrn" => ytrn,
						"ytst" => ytst))
		end
		#Set percentage of y-data collected
		idx_hide = randsubseq(1:size(xtrn,4), drop_p)
		ytrn[:,idx_hide] .= nothing
	end
    return (xtrn, xtst, ytrn, ytst)
end
export loaddata


function defaultFE_w(resfile)
	res_model = matopen(resfile)
	params = read(res_model, "params")
	atype = Array{Float32}

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
	push!(ws, [0.01])
    w = map(wi->convert(atype, wi), ws)
	w[1] = w[1][:,:,1:2,:]
	return w
end


"""
    defaultFE(w, x; noiseSize=0.005, mode=true)

Default Front End Network to reduce sample images to feature vector.
Uses ResNet50 architecture to generate 1000-length vector.

<b> Arguments </b>
- `w::Array`: Array of neural network weights stored as KnetArray or Array.
- `x::Array`: Array of samples (real or fake).
- `noiseSize::Float`: St. deviation of normal used to add gaussian noise to sample.
- `mode::bool`:  batch normalization train/test mode.
"""

function defaultFE(w, x; noiseSize=0.005, mode=true, atype=Array{Float32})
	noise =  convert(atype, randn(size(x)) * noiseSize)
	x = x + noise #adding some artificial noise

	conv1  = conv4(w[1],x; padding=3, stride=2) .+ w[2]
    bn1    = batchnorm(conv1, bnmoments(); training=mode)
    pool1  = pool(conv1; window=3, stride=2)

    # layer 2,3,4,5
    r2 = reslayerx5(w[5:34], pool1, w[163]; strides=[1,1,1,1], mode=mode)
    r3 = reslayerx5(w[35:73], r2, w[163]; mode=mode)
    r4 = reslayerx5(w[74:130], r3, w[163]; mode=mode) # 5
    r5 = reslayerx5(w[131:160], r4, w[163]; mode=mode)

    # fully connected layer
    pool5  = pool(r5; stride=1, window=2, mode=2)
    fc1000 = w[161] * mat(pool5) .+ w[162]
end
function reslayerx0(w, x; padding=0, stride=1, mode=true)
    b  = conv4(w[1], x; padding=padding, stride=stride)
    bx = batchnorm(b, bnmoments(); training=mode)
end
function reslayerx1(w, x, p; padding=0, stride=1, mode=0)
    prelu(reslayerx0(w, x; padding=padding, stride=stride, mode=mode), p)
end
function reslayerx2(w,x,p; pads=[0,1,0], strides=[1,1,1], mode=0)
    ba = reslayerx1(w[1:3],x, p; padding=pads[1], stride=strides[1], mode=mode)
    bb = reslayerx1(w[4:6],ba, p; padding=pads[2], stride=strides[2], mode=mode)
    bc = reslayerx0(w[7:9],bb; padding=pads[3], stride=strides[3], mode=mode)
end
function reslayerx3(w,x,p; pads=[0,0,1,0], strides=[2,2,1,1], mode=0) # 12
    a = reslayerx0(w[1:3],x; stride=strides[1], padding=pads[1], mode=mode)
    b = reslayerx2(w[4:12],x,p; strides=strides[2:4], pads=pads[2:4], mode=mode)
    prelu(a .+ b, p)
end
function reslayerx4(w,x,p; pads=[0,1,0], strides=[1,1,1], mode=0)
    prelu(x .+ reslayerx2(w,x,p; pads=pads, strides=strides, mode=mode),p)
end
function reslayerx5(w,x,p; strides=[2,2,1,1], mode=0)
    x = reslayerx3(w[1:12],x,p; strides=strides, mode=mode)
    for k = 13:9:length(w)
        x = reslayerx4(w[k:k+8],x,p; mode=mode)
    end
    return x
end
export defaultFE_w, defaultFE


function defaultD_w(fc_length)
	w = Array{Any}(undef, 2)
	w[1] = randn(1, fc_length) * sqrt(2/fc_length) #μ=0, σ=sqrt(2/1000)
	w[2] = zeros(1,1)
	return w
end

"""
    defaultD(wD, fc)

Default Discriminator Network to provide scalar prediction of "realness" of sample.

<b> Arguments </b>
- `wD::Array`: Array of neural network weights stored as KnetArray or Array.
- `fc::Array`: array of features generated by Front-End Network.
"""
function defaultD(wD, fc)
	return (wD[1] * fc) .+ wD[2]
end
export defaultD_w, defaultD


function defaultQ_w(disc_ss, disc, cont, fc_length)
	c_size = length(disc_ss) + length(disc) + (cont>0 ? 1 : 0)
	w_size = 2 + c_size*2
	w = Array{Any}(undef, w_size)

	#hidden layer
	w[1:2] = [randn(128, fc_length) * sqrt(2/fc_length), zeros(128,1)]

	#discrete scores semisupervised
	for i = 1:length(disc_ss)
		k = disc_ss[i]
		w[2i-1+2 : 2i+2] = [randn(k, 128) * sqrt(2/128), zeros(k,1)]
	end

	#discrete scores
	start = 2 + (2*length(disc_ss))
	for i = 1:length(disc)
		k = disc[i]
		w[2i-1+start:2i+start] = [randn(k, 128) * sqrt(2/128), zeros(k,1)]
	end

	#mu for continuous
	cont>0 && (w[w_size-1:w_size] = [randn(cont*2, 128) * sqrt(2/128), zeros(cont*2, 1)])
	return w
end

"""
    defaultQ(wQ, fc, numDiscSS, numDisc, numCont; mode=mode)

Default Auxiliary Network to predict latent code distributions.

<b> Arguments </b>
- `wQ::Array`: Array of neural network weights stored as KnetArray or Array.
- `fc::Array`: array of features generated by Front-End Network.
- `numDiscSS::Int`: Number of semisupervised discrete codes.
- `numDisc::Int`: Number of unsupervised discrete codes.
- `numCont::Int`: Number of continuous codes.
- `mode::bool`:  batch normalization train/test mode.
"""
function defaultQ(wQ, fc, numDiscSS, numDisc, numCont; mode=true)
	fc2 = bn_relu((wQ[1] * fc) .+ wQ[2], mode)

	numDiscSS>0 ? (logitsSS = getlogits(wQ[3:end], numDiscSS, fc2)) : logitsSS=nothing
	numDisc>0 ? (logits = getlogits(wQ[3+(numDiscSS*2):end], numDisc, fc2)) : logits=nothing

	mu, var = nothing, nothing
	if numCont > 0
		W_cont, b_cont = wQ[end-1:end]
		fc_cont = (W_cont * fc2) .+ b_cont
		mu = fc_cont[1:numCont,:]
		var = exp.(fc_cont[numCont+1:end,:])
	end
	return logitsSS, logits, mu, var
end
export defaultQ_w, defaultQ
function getlogits(wQ, numDisc, fc)
	logits = Array{Any}(undef, numDisc)
	for i = 1:numDisc
		logits[i] = (wQ[2*i-1] * fc) .+ wQ[2*i]
	end
	return logits
end


function defaultG_w(codeSize)
	#Yi = Wi+stride[i](Xi-1)-2padding[i]
	depths = [codeSize, 1024, 512, 512, 128, 128, 64, 64]
	upsample_fs = [2, 2, 2, 2, 2, 2] #4*2*2*2*2 = 64
	N_db = 1; N_drb=length(upsample_fs) - N_db
	db_s = 4; drb_s = 2 * db_s + 2

	numWeights = 2 + N_db*db_s + N_drb*drb_s + 2 + 1

	# Linear Layer
	w = Array{Any}(undef, numWeights)
	w[1:2] = [randn(1,1,depths[1],depths[2]), zeros(1,1,depths[2],1)]

	# Deconv Layers
	# Regular Deconvolutions
	for i = 1:N_db
		w[2 + db_s*(i - 1) + 1:2 + db_s*i] = deconv_block_w(upsample_fs[i], depths[i+1], depths[i+2])
	end
	# Residual Deconvolutions
	start = db_s*N_db + 2
	for i = 1:N_drb
		w[drb_s*(i - 1) + 1 + start:drb_s*i + start] = deconv_resblock_w(upsample_fs[N_db + i], depths[i+N_db+1], depths[i+N_db+2])
	end

	# Final Convolution
	final_depth = 2
	w[numWeights-2: numWeights-1] = [randn(1,1,depths[end],final_depth), zeros(1,1,final_depth,1)]

	# Leaky ReLu parameter
	w[numWeights] = [0.01]
	return w
end


"""
    defaultG(wG, Z; mode=true)

Default Generator Network to create image samples from latent code.


<b> Arguments </b>
- `wG::Array`: Array of neural network weights stored as KnetArray or Array.
- `Z::Array`: latent code.
- `mode::bool`:  batch normalization train/test mode.
"""
function defaultG(wG, Z; mode=true)
	# #Linear Layer
	g1 = bn_prelu(conv4(wG[1], Z) .+ wG[2], mode, wG[end])

	upsample_fs = [2, 2, 2, 2, 2, 2] #4*2*2*2*2 = 64
	N_db = 1; N_drb=length(upsample_fs) - N_db
	db_s = 4; drb_s = 2 * db_s + 2

	# Rgular Deconvolution Blocks
	for i = 1:N_db
		g1 = deconv_block(wG[2 + db_s*(i - 1) + 1:2 + db_s*i], g1)
		g1 = bn_prelu(g1, mode, wG[end])
	end
	# Residual Deconvolution Blocks
	start = db_s * N_db + 2
	for i = 1:N_drb
		g1 = deconv_resblock(wG[drb_s*(i - 1) + 1 + start:drb_s*i + start], g1, wG[end]; mode=mode)
		g1 = bn_prelu(g1, mode, wG[end])
	end

	# Final Conv
	g2 = conv4(wG[end-2], g1; padding=0) .+ wG[end-1]
	return tanh.(g2)
end
export defaultG_w, defaultG

function deconv_block(w, x; mode=true, resize=false)
	#Yi = Wi+stride[i](Xi-1)-2padding[i]
	g1 = deconv4(w[1], x; stride=size(w[1], 1)) .+ w[2]
	return conv4(w[3], g1; padding=1) .+ w[4]
	# g1 = bn_prelu(, mode, p)
	# return bn_prelu(conv4(w[5], g1; padding=1) .+ w[6], mode, p)
end
function deconv_block_w(f, inC, outC)
	w = Array{Any}(undef, 4)
	w[1] = randn(f, f, outC, inC) * sqrt(2/(inC*f*f))
	w[2] = zeros(1,1,outC,1)
	w[3] = randn(3,3,outC,outC) * sqrt(2/(outC*9))
	w[4] = zeros(1,1,outC,1)
	# w[5] = randn(3,3,inC,outC) * sqrt(2/(outC*9))
	# w[6] = randn(1,1,outC,1)
	return w
end
function deconv_resblock(w, x, p; mode=true)
	# Branch 1
	a = deconv_block(w[1:4], x)

	# Branch 2
	b = bn_prelu(conv4(w[5], x; padding=1) .+ w[6], mode, p)
	b = deconv_block(w[7:10], b)
	return a + b
end
function deconv_resblock_w(f, inC, outC)
	w = Array{Any}(undef, 10)
	w[1:4] = deconv_block_w(f, inC, outC)
	w[5:6] = [randn(3,3,inC,inC) * sqrt(2/(inC*9)), randn(1,1,inC,1)]
	w[7:10] = deconv_block_w(f, inC, outC)
	return w
end

""" wrapped batch normalization function """
bn(h, mode) = batchnorm(h, bnmoments(); training=mode)
bn_relu(h, mode) = relu.(bn(h, mode))
prelu(x, p) = max.(p .* x, x)
bn_prelu(h, mode, p) = prelu(bn(h, mode), p)

end
