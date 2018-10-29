#InfoGAN_Fns.jl
module InfoGAN_Fns
using helpers
using Compat, Pkg, Distributions, MAT, Knet, Printf, Images, Random, ArgParse

import Knet: size
function size(a::KnetArray{T,N}, t::Tuple) where {T,N}
	x =size(a); s = []; for i = t; i>N ? push!(s,1) : push!(s, x[i]) ; end
	return s
end

export FrontEnd, Discriminator, Auxiliary, Generator

mutable struct FrontEnd
	w::AbstractArray
	atype::Union{Type{KnetArray{Float32}}, Type{Array{Float32}}}
	function FrontEnd(o::Dict)
		res_model = matopen("imagenet-resnet-50-dag.mat")
		w, _ = get_params(read(res_model, "params"), Array{Float32})
		# Change channels:
		w[1] = w[1][:,:,1:1,:]
		w = map(x->convert(o[:atype], x), w)
		@info "Initialized FrontEnd weights:" size(w) typeof(w)
		return new(w, o[:atype])
	end
end
function frontend_(wF, x, f::FrontEnd; noiseSize=0.001, mode=true)
	noise =  convert(f.atype, randn(size(x)) * noiseSize)
	x = tanh.(x + noise) #adding some artificial noise
	# Res-Net to get 1000-Feature Vector
	return resnet50(wF, x; mode=mode)
end


mutable struct Discriminator
	w::AbstractArray;
	atype::Union{Type{KnetArray{Float32}}, Type{Array{Float32}}}
	function Discriminator(o::Dict)
		layerNum = 1
		w = Array{Any}(undef, layerNum * 2)
		w[1] = randn(1, 1000) * sqrt(2/1000) #μ=0, σ=sqrt(2/1000)
		w[2] = zeros(1,1)

		w = map(x->convert(o[:atype], x), w)
		@info "Initialized Discriminator weights:" size(w) typeof(w)
		return new(w, o[:atype])
	end
end
function discriminator_(wD, fc, d::Discriminator)
	W1, b1 = wD
	p = sigm.((W1 * fc) .+ b1) #Should be size 2 x N
	return p
end


mutable struct Auxiliary
	w::AbstractArray;
	atype::Union{Type{KnetArray{Float32}}, Type{Array{Float32}}}
	numDisc::Int64
	numCont::Int64
	function Auxiliary(o::Dict)
		layerNum = 1 + length(o[:c_disc]) + 1
		w = Array{Any}(undef, layerNum * 2)

		w[1:2] = [randn(128, 1000) * sqrt(2/1000), zeros(128,1)]
		#discrete scores
		i = 3
		for k = o[:c_disc]
		  w[i:i+1] = [randn(k, 128) * sqrt(2/128), zeros(k,1)]; i+=2
		end
		#mu for continuous
		if o[:c_cont] > 0
			w[i:i+1] = [randn(o[:c_cont]*2, 128) * sqrt(2/128), zeros(o[:c_cont]*2, 1)]
		end
		w = map(x->convert(o[:atype], x), w)
		@info "Initialized Auxiliary weights:" size(w) typeof(w)
		return new(w, o[:atype], length(o[:c_disc]), o[:c_cont])
	end
end
function auxiliary_(wQ, fc, q::Auxiliary; mode=true)
	W1, b1 = wQ[1:2]
	fc2 = relu.(batchnorm((W1 * fc) .+ b1, bnmoments(); training=mode))

	logits = Array{Any}(undef, q.numDisc)
	for i = 1:q.numDisc
		logits[i] = (wQ[2*i+1] * fc2) .+ wQ[2*i+2]
	end
	if q.numCont > 0
		W_cont, b_cont = wQ[end-1:end]
		fc_cont = (W_cont * fc2) .+ b_cont
		mu = fc_cont[1:q.numCont,:]
		var = exp.(fc_cont[q.numCont+1:end,:])
		return logits, mu, var
	else
		return logits, nothing, nothing
	end
end

mutable struct Generator
	w::AbstractArray
	atype::Union{Type{KnetArray{Float32}}, Type{Array{Float32}}}
	z_size::Int64
	c_disc::Union{Tuple, Int64}
	c_cont::Int64
	bs::Int64
	function Generator(o::Dict)
		#TODO Implement with depths as function argument inputs
		codeSize = sum(o[:c_disc]) + o[:c_cont] + o[:z]
		depths = 	[codeSize, 1024, 512, 128, 64, 1]
		windows = 	[1, 7, 4, 4, 4]
		layerNum = length(windows)
		w = Array{Any}(undef, layerNum*2)

		for i = 1:layerNum
			s = (windows[i], windows[i], depths[i+1], depths[i])
			w[(2*i) - 1] = randn(s...) * sqrt(2/depths[i])
			w[2*i] = zeros(depths[i+1], 1)
		end

		w = map(x->convert(o[:atype], x), w)
		@info "Initialized Generator weights: " size(w) typeof(w)
		return new(w, o[:atype], o[:z], o[:c_disc], o[:c_cont], o[:batchsize])
	end
end
function generator_(wG, Z, g::Generator; mode=true)
	p_drop = 0.5
	strides = [1, 1, 2, 2, 2]
	paddings = [0, 0, 0, 1, 1]
	layerNum = length(strides)

	prev_input = Z
	for i = 1:layerNum
		dconv = deconv4(wG[2i-1], prev_input; padding=paddings[i], stride=strides[i], mode=1)
		i<5 ? dconv = relu.(batchnorm(dconv, bnmoments(); training=mode)) : dconv = tanh.(dconv)
		prev_input = dconv
	end
	return prev_input
end

function trainD!(F, D, G, xs, Dopt, Fopt)
	#Get gradient function wrt to first argument (Dnet, ie. weights)
	D_gradfun = grad(D_loss_, 1)
	F_gradfun = grad(D_loss_, 2)
	δD = D_gradfun(D.w, F.w, xs, D, F)
	δF= F_gradfun(D.w, F.w, xs, D, F)

	for i=1:length(D.w)
		update!(D.w[i], δD[i], Dopt[i])
	end
	for i=1:length(F.w)
		update!(F.w[i], δF[i], Fopt[i])
	end
end
function D_loss(F, D, G, x; mode=true, v=false)
	fake_x = generator_(G.w, samplenoise(G)[1], G; mode=mode)
	return D_loss_(D.w, F.w, [x, fake_x], D, F; v=v), [x, fake_x]
end
function D_loss_(wD, wF, xs, D, F; ep = 1e-15, v=false)
	r_fc = frontend_(wF, xs[1], F; mode=mode)
	f_fc = frontend_(wF, xs[2], F; mode=mode)

	r_loss = -log.(discriminator_(wD, r_fc, D) .+ ep) 		#D(x)=1 => loss = 0
	f_loss = -log.(1 .- discriminator_(wD, f_fc, D) .+ ep) 	#D(G(z)))=0 => loss = 0

	loss = mean(r_loss .+ f_loss)
	v && @printf "\tBatch 1 Real Loss: %.3f, and Fake Loss: %.3f\n" mean(r_loss) mean(f_loss);
	return loss
end
export trainD!, D_loss


function trainG!(F, D, G, Q, Zinfo, Gopt, Qopt)
	G_gradfun = grad(G_loss_, 1)
	Q_gradfun = grad(G_loss_, 2)
	δG = G_gradfun(G.w, Q.w, Zinfo, F, D, G, Q)
	δQ= Q_gradfun(G.w, Q.w, Zinfo, F, D, G, Q)

	for i=1:length(G.w)
		update!(G.w[i], δG[i], Gopt[i])
    end
	for i=1:length(Q.w)
		update!(Q.w[i], δQ[i], Qopt[i])
	end
end
#generator loss
function G_loss(F, D, G, Q; mode=true, v=false)
	#Create Fake Samples
	Zinfo = samplenoise(G)
	loss = G_loss_(G.w, Q.w, Zinfo, F, D, G, Q; v=v)
	return loss, Zinfo
end
export G_loss
function G_loss_(wG, wQ, Zinfo, F, D, G, Q; ep=1e-15, v=false)
	Z, idxes = Zinfo
	loss = 0

	fake_fc = frontend_(F.w, generator_(wG, Z, G), F)
	g_loss = mean(-log.(discriminator_(D.w, fake_fc, D) .+ ep));
	loss += g_loss

	d_loss=0; logits, μ, σ = auxiliary_(wQ, fake_fc, Q)
	for i=1:length(idxes); d_loss += nll(logits[i], idxes[i]; dims=1); end;
	loss += d_loss

	i = sum(Q.numDisc) + 1; j = i + Q.numCont - 1
	c_loss = log_gaussian(mat(Z)[i:j, :], μ, σ) * 0.05;
	loss+= c_loss

	v && @printf "\tBatch 1 Gen Loss: %.3f, Disc Loss: %.3f, and Cont Loss: %.3f\n" g_loss d_loss c_loss
	return loss
end
export trainG!, G_loss


#auxiliary loss
function log_gaussian(z, μ, σ; ϵ=1e-15)
	loglikelihood = -0.5 .* log.(σ .* 2 .* π .+ ϵ) .- (abs2.(μ-z) ./ (2 .* σ .+ ϵ))
	size(loglikelihood, 1) < 2 ? s = -loglikelihood : (println("!!!!"); s = sum(-loglikelihood; dims=1))
	return mean(s)
	# return mean(sum(-1 .* loglikelihood; dims=1))
end


function samplenoise(c_disc::Union{Tuple, Int}, c_cont::Int, z_size::Int, bs::Int, atype)
	#TODO Implement with preallocated array !
	idxes = []
	cs = []
	for k = c_disc
		idx = rand(1:k, bs)
		c = zeros(Int, 1, 1, k, bs);
		for i = 1:bs; c[1,1,idx[i], i] = 1; end
		push!(idxes, idx)
		push!(cs, c)
	end
	# c_disc = cat(cs...; dims=3)
	c_cont = (rand(1, 1, c_cont, bs) .* 2) .-1
	noise = (rand(1, 1, z_size, bs) .* 2) .-1
	return convert(atype, cat(cs..., c_cont, noise; dims=3)), idxes
end
samplenoise(G::Generator) = samplenoise(G.c_disc, G.c_cont, G.z_size, G.bs, G.atype)
samplenoise(G::Generator, gencnt::Int) = samplenoise(G.c_disc, G.c_cont, G.z_size, gencnt, G.atype)
export samplenoise


function print_output(epoch, G::Generator, gencnt::Int)
	Z, _ = samplenoise(G, gencnt)
	G_sample = convert(Array{Float32}, generator_(G.w, Z, G; mode=false))
	out = map(clamp01nan, Gray.(makegrid(G_sample)))

	savefile = @sprintf "outputs/epoch%d.png" epoch
	save(savefile, out)
end
export print_output


mutable struct InfoModel
	F::FrontEnd
	D::Discriminator
	G::Generator
	Q::Auxiliary
	Fopt::AbstractArray
	Dopt::AbstractArray
	Gopt::AbstractArray
	Qopt::AbstractArray
	function InfoModel(o)
		F = FrontEnd(o);
		D = Discriminator(o);
		G = Generator(o);
		Q = Auxiliary(o);

		#ADAM Optmizer with initial LR=0.0002, and beta1=0.5
		Fopt = map(x->Adam(;lr=o[:lr][1], beta1=0.5), F.w)
		Dopt = map(x->Adam(;lr=o[:lr][2], beta1=0.5), D.w)
		Gopt = map(x->Adam(;lr=o[:lr][3], beta1=0.5), G.w)
		Qopt = map(x->Adam(;lr=o[:lr][4], beta1=0.5), Q.w)
		return new(F, D, G, Q, Fopt, Dopt, Gopt, Qopt)
	end
end
export InfoModel



function train(xtrn, ytrn, model, o; mdlfile=nothing, logfile=nothing)
	fh = open(logfile,"w")

	traindata = Array{Float64}(undef, o[:epochs], 2)

	for epoch = 1:o[:epochs]
		epoch%2==0 && o[:v] ? verbose=true : verbose=false

		trnIdx = mbatch(randperm(size(xtrn)[end]), o[:batchsize])
		lossD, lossG = 0, 0
		total = length(trnIdx)
		for i = 1:total
			verbose && i == 1 ? batchv = true : batchv = false
			x = convert(model.F.atype, xtrn[:, :, :, trnIdx[i]])

			#train D
			for j=1:o[:dreps]
				batch_lossD, xs = D_loss(model.F, model.D, model.G, x; mode=true, v=(batchv && j==o[:dreps]))
				trainD!(model.F, model.D, model.G, xs, model.Dopt, model.Fopt)
				lossD += batch_lossD
		    end

		    #train G
		    for k=1:o[:greps]
				batch_lossG, Zinfo = G_loss(model.F, model.D, model.G, model.Q; mode=true, v=batchv)
				trainG!(model.F, model.D, model.G, model.Q, Zinfo, model.Gopt, model.Qopt)
				lossG += batch_lossG
		    end
		end
		lossG /= total; lossD /= total;
		verbose && @printf("epoch: %d loss[D]: %g loss[G]: %g\n", epoch, lossD, lossG)
		write(fh, "epoch: $epoch loss[D]: $lossD loss[G]: $lossG\n")
		traindata[epoch, 1:2] = [lossD, lossG]

		epoch%2==0 && print_output(epoch, model.G, o[:gencnt])
		epoch%2==0 && save_weights(model; savefile=mdlfile)
	end
	close(fh)
	return traindata
end
export train
end
