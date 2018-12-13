#InfoGAN_Fns.jl
module InfoGAN_Fns
using helpers: defaultFE_w, defaultFE, defaultD_w, defaultD, defaultG_w, defaultG, defaultQ_w, defaultQ
using Knet, Printf, Random, JLD2, Statistics, Images

import Knet: size
function size(a::KnetArray{T,N}, t::Tuple) where {T,N}
	x =size(a); s = []; for i = t; i>N ? push!(s,1) : push!(s, x[i]) ; end
	return s
end

export FrontEnd, Discriminator, Auxiliary, Generator

mutable struct FrontEnd
	w::AbstractArray
	atype
	forward
	opt::AbstractArray
	function FrontEnd(w, atype, forward, opt)
		s = @sprintf "Number of Parameters: %.3f M\n" sum(map(x->prod(size(x)), w)) / 1e6
		@info "Initialized FrontEnd weights:" size(w) typeof(w) s
		return new(w, atype, forward, opt)
	end
	function FrontEnd(w::AbstractArray, o::Dict, forward)
		Fopt = map(x->Rmsprop(;lr=o[:lr][1]), w)
		return FrontEnd(w, o[:atype], forward, Fopt)
	end
function FrontEnd(o::Dict)
		w = defaultFE_w("../resources/resnet.mat")
		w = map(x->convert(o[:atype], x), w)
		return FrontEnd(w, o, defaultFE)
	end
end

mutable struct Discriminator
	w::AbstractArray;
	atype
	forward
	opt::AbstractArray
	function Discriminator(w, atype, forward, opt)
		s = @sprintf "Number of Parameters: %.3f M\n" sum(map(x->prod(size(x)), w)) / 1e6
		@info "Initialized Discriminator weights:" size(w) typeof(w) s
		return new(w, atype, forward, opt)
	end
	function Discriminator(w, o::Dict, forward)
		Dopt = map(x->Rmsprop(;lr=o[:lr][2]), w)
		return Discriminator(w, o[:atype], forward, Dopt)
	end
	function Discriminator(o::Dict)
		w = defaultD_w(1000) #resnet Fc length
		w = map(x->convert(o[:atype], x), w)
		return Discriminator(w, o, defaultD)
	end
end

mutable struct Generator
	w::AbstractArray
	atype
	forward
	opt::AbstractArray
	function Generator(w, atype, forward, opt)
		s = @sprintf "Number of Parameters: %.3f M\n" sum(map(x->prod(size(x)), w)) / 1e6
		@info "Initialized Generator weights:" size(w) typeof(w) s
		return new(w, atype, forward, opt)
	end
	function Generator(w::AbstractArray, o::Dict, forward)
		Gopt = map(x->Rmsprop(;lr=o[:lr][3]), w)
		return Generator(w, o[:atype], forward, Gopt)
	end
	function Generator(o::Dict)
		codeSize = sum(o[:c_SS]) + sum(o[:c_disc]) + o[:c_cont] + o[:z]
		w = defaultG_w(codeSize)
		w = map(x->convert(o[:atype], x), w)
		@info "Initialized Generator weights: " size(w) typeof(w)
		return Generator(w, o, defaultG)
	end
end

mutable struct Auxiliary
	w::AbstractArray;
	atype
	forward
	opt::AbstractArray
	function Auxiliary(w, atype, forward, opt)
		s = @sprintf "Number of Parameters: %.3f M\n" sum(map(x->prod(size(x)), w)) / 1e6
		@info "Initialized Auxiliary weights:" size(w) typeof(w) s
		return new(w, atype, forward, opt)
	end
	function Auxiliary(w::AbstractArray, o::Dict, forward)
		Qopt = map(x->Rmsprop(;lr=o[:lr][4]), w)
		return Auxiliary(w, o[:atype], forward, Qopt)
	end
	function Auxiliary(o::Dict)
		w = defaultQ_w(o[:c_SS], o[:c_disc], o[:c_cont], 1000)
		w = map(x->convert(o[:atype], x), w)
		function auxiliary_(wQ, fc, o::Dict; mode=true)
			return defaultQ(wQ, fc, length(o[:c_SS]), length(o[:c_disc]), o[:c_cont]; mode=mode)
		end
		return Auxiliary(w, o, auxiliary_)
	end
end

mutable struct InfoModel
	F::FrontEnd
	D::Discriminator
	G::Generator
	Q::Auxiliary
	o::Dict
	InfoModel(F, D, G, Q, o) = new(F, D, G, Q, o)
	function InfoModel(o::Dict)
		F = FrontEnd(o);
		D = Discriminator(o);
		G = Generator(o);
		Q = Auxiliary(o);
		return new(F, D, G, Q, o)
	end
end
export InfoModel

function save_model(model::InfoModel; savefile="model.jld2")
	save(savefile, Dict("F" => [map(x-> convert(Array{Float32}, x), model.F.w)],
						"D" => [map(x-> convert(Array{Float32}, x), model.D.w)],
						"G" => [map(x-> convert(Array{Float32}, x), model.G.w)],
						"Q" => [map(x-> convert(Array{Float32}, x), model.Q.w)],
						"o" => model.o,))
end
function load_model(mdlfile)
	m = load(mdlfile)
	o = m["o"]
	if typeof(o[:atype].body.name) <: String
	# println(typeof(o[:atype].body.name))
		o[:atype] = (occursin("Knet", o[:atype].body.name) ? "KnetArray{Float32}" : "Array{Float32}")
		o[:atype] = eval(Meta.parse(o[:atype]))
	end

	function frontend_(wF, x, f::FrontEnd; noiseSize=0.005, mode=true)
		noise =  convert(f.atype, randn(size(x)) * noiseSize)
		x = x + noise #adding some artificial noise
		# Res-Net to get 1000-Feature Vector
		return defaultFE(wF, x; mode=mode)
	end

	function auxiliary_(wQ, fc, o::Dict; mode=true)
		return defaultQ(wQ, fc, length(o[:c_SS]), length(o[:c_disc]), o[:c_cont]; mode=mode)
	end

	F = FrontEnd(map(x-> convert(KnetArray{Float32}, x), m["F"][1]), o, frontend_)
	D = Discriminator(map(x-> convert(KnetArray{Float32}, x), m["D"][1]), o, defaultD)
	G = Generator(map(x-> convert(KnetArray{Float32}, x), m["G"][1]), o, defaultG)
	Q = Auxiliary(map(x-> convert(KnetArray{Float32}, x), m["Q"][1]), o, auxiliary_)
	return InfoModel(F, D, G, Q, o)
end
export save_model, load_model


function trainD!(m::InfoModel, xs)
	#Get gradient function wrt to first argument (Dnet, ie. weights)
	D_gradfun = grad(D_loss_, 1)
	F_gradfun = grad(D_loss_, 2)
	δD = D_gradfun(m.D.w, m.F.w, xs, m)
	δF= F_gradfun(m.D.w, m.F.w, xs, m)

	for i=1:length(m.D.w)
		update!(m.D.w[i], δD[i], m.D.opt[i])
		#Gradient Clipping
		m.D.w[i] = min.(m.D.w[i], m.o[:clip])
		m.D.w[i] = max.(m.D.w[i], m.o[:clip])
	end
	for i=1:length(m.F.w)
		update!(m.F.w[i], δF[i], m.F.opt[i])
		#Gradient Clipping
		m.F.w[i] = min.(m.F.w[i], m.o[:clip])
		m.F.w[i] = max.(m.F.w[i], m.o[:clip])
	end
end

function D_loss(m::InfoModel, x::Union{KnetArray, Array}; mode=true, v=false)
	fake_x = m.G.forward(m.G.w, samplenoise(m, size(x,4))[1]; mode=mode)
	return D_loss_(m.D.w, m.F.w, [x, fake_x], m; v=v, mode=mode), [x, fake_x]
end
function D_loss_(wD, wF, xs, m::InfoModel; ep = 1e-15, v=false, mode=true)
	r_fc = m.F.forward(wF, xs[1]; mode=mode)
	f_fc = m.F.forward(wF, xs[2]; mode=mode)

	# Traditional Loss Function (GAN)
	# r_loss = -log.(m.D.forward(wD, r_fc) .+ ep) 		#D(x)=1 => loss = 0
	# f_loss = -log.(1 .- m.D.forward(wD, f_fc) .+ ep) 	#D(G(z)))=0 => loss = 0

	#Wassertein loss
	d_loss = mean(-m.D.forward(wD, r_fc)) + mean(m.D.forward(wD, f_fc))

	# Gradient Penalty Method for Weight Regularization
	# Cannot be implemented until we figure out how to do
	# d/dW (d/dx D(x))

	# bs = size(xs[1],4)
	# ϵ = KnetArray{Float32}(rand(1,1,1,bs))
    # x_hat = (ϵ .* xs[1]) .+ ((1 .- ϵ) .* xs[2])
	# ∇D_x_hat = grad(x->sum(m.D.forward(wD, m.F.forward(wF, x; mode=mode))), 1)(x_hat)
    # ∇D_norms = sqrt.(eps() .+ sum(∇D_x_hat .* ∇D_x_hat;dims=(1,2,3)))
	# println("norm mean: ", mean(∇D_norms))
	# gp_loss = mean(m.o[:gpCoeff] *(∇D_norms .-1).^2)

	gp_loss = 0
	loss = d_loss + gp_loss
	v && @printf "\tBatch Discriminator Loss: %.3e (D: %.3e, GP: %.3e)\n" loss d_loss gp_loss;
	return loss
end
export trainD!, D_loss
# gp_fun(x_hat, wF, wD, m::InfoModel, mode) = sum(m.D.forward(wD, m.F.forward(wF, x_hat; mode=mode)))
# gp_grad = grad(gp_fun, 1)


function trainG!(m::InfoModel, Zinfo, x::Union{KnetArray, Array}, y::Union{KnetArray, Array})
	G_gradfun = grad(G_loss_, 1)
	Q_gradfun = grad(G_loss_, 2)
	δG = G_gradfun(m.G.w, m.Q.w, Zinfo, x, y, m)
	δQ= Q_gradfun(m.G.w, m.Q.w, Zinfo, x, y, m)

	for i=1:length(m.G.w)
		update!(m.G.w[i], δG[i], m.G.opt[i])
    end
	for i=1:length(m.Q.w)
		update!(m.Q.w[i], δQ[i], m.Q.opt[i])
	end
end
#generator loss
function G_loss(m::InfoModel, x::Union{KnetArray, Array}, y::Union{KnetArray, Array}; mode=true, v=false)
	#Create Fake Samples
	Zinfo = samplenoise(m)
	loss = G_loss_(m.G.w, m.Q.w, Zinfo, x, y, m; mode=mode, v=v)
	return loss, Zinfo
end
export G_loss
function G_loss_(wG, wQ, Zinfo, x::Union{KnetArray, Array}, y::Union{KnetArray, Array}, m::InfoModel; ep=1e-15, v=false, mode=true)
	Z, idxes = Zinfo
	fake_fc = m.F.forward(m.F.w, m.G.forward(wG, Z; mode=mode))
	fc = m.F.forward(m.F.w, x)

	_, logits, μ, σ = m.Q.forward(wQ, fake_fc, m.o)
	logitsSS, _, _, _ = m.Q.forward(wQ, fc, m.o)

	loss = 0
	# Traditional GAN loss
	# g_loss = mean(-log.(m.D.forward(m.D.w, fake_fc) .+ ep));

	# Wasserstein loss
	g_loss = mean(-m.D.forward(m.D.w, fake_fc))
	loss += g_loss

	d_lossSS = 0
	for i=1:length(m.o[:c_SS])
		idx_lbl = y[i,:] .!= nothing
		~(any(idx_lbl)) && continue
		d_lossSS += nll(logitsSS[i][:,idx_lbl], convert(Array{Int64,1}, y[i, idx_lbl]); dims=1)
	end
	loss += d_lossSS * m.o[:discCoeffSS]

	d_loss=0
	for i=1:length(idxes); d_loss += nll(logits[i], idxes[i]; dims=1); end;
	loss += d_loss * m.o[:discCoeff]

	if m.o[:c_cont] > 0
		i = sum(m.o[:c_SS]) + sum(m.o[:c_disc]) + 1
		j = i + m.o[:c_cont] - 1
		c_loss = log_gaussian(mat(Z)[i:j, :], μ, σ) * 0.05; loss += c_loss
		v && @printf "\tBatch1 L-Gen: %.3e, L-DiscSS: %.3f, L-Disc: %.3f, and L-Cont: %.3f\n\n" g_loss d_lossSS d_loss c_loss
	else
		v && @printf "\tBatch1 L-Gen: %.3e, L-Disc: %.3f\n\n" g_loss d_loss
	end
	return loss
end
export trainG!, G_loss

#auxiliary loss
function log_gaussian(z, μ, σ; ϵ=1e-15)
	loglikelihood = -0.5 .* log.(σ .* 2 .* π .+ ϵ) .- (abs2.(μ-z) ./ (2 .* σ .+ ϵ))
	size(loglikelihood, 1) < 2 ? s = -loglikelihood : (s = sum(-loglikelihood; dims=1))
	return mean(s)
end

function samplenoise(o::Dict, numSamples)
	c_SS, c_disc, nC, nZ = o[:c_SS], o[:c_disc], o[:c_cont], o[:z]
	codeSize = sum(c_SS) + sum(c_disc) + nC + nZ
	z = Array{Any}(undef, 1, 1, codeSize, numSamples)

	for k = 1:length(c_SS)
		j = sum(c_SS[1:k]); i = j-c_SS[k]+1;
		idx = rand(1:c_SS[k], numSamples)
		z[1,1,i:j,:] = zeros(Int, 1, 1, c_SS[k], numSamples)
		for n = 1:numSamples; z[1,1,idx[n]+i-1, n] = 1; end
	end

	idxes = Array{Any}(undef, length(c_disc))
	for k = 1:length(c_disc)
		j = sum(c_SS) + sum(c_disc[1:k]); i = j-c_disc[k]+1;
		idx = rand(1:c_disc[k], numSamples)
		z[1,1,i:j,:] = zeros(Int, 1, 1, c_disc[k], numSamples)
		for n = 1:numSamples; z[1,1,idx[n]+i-1, n] = 1; end
		idxes[k] = idx
	end
	# c_disc = cat(cs...; dims=3)
	i = sum(c_SS) + sum(c_disc) + 1; j = i + nC - 1
	z[1,1,i:j,:] = (rand(1, 1, nC, numSamples) .* 2) .-1
	z[1,1,end-nZ+1:end,:] = (rand(1, 1, nZ, numSamples) .* 2) .-1
	return convert(o[:atype], z), idxes
end
samplenoise(M::InfoModel, cnt::Int) = samplenoise(M.o, cnt)
samplenoise(M::InfoModel) = samplenoise(M.o, M.o[:batchsize])
export samplenoise


function print_output(epoch, m::InfoModel, s::Int, scale::Float64, gs::Int;printfolder="outputs")
	Z, _ = samplenoise(m, m.o[:printNum])
	G_sample = convert(Array{Float32}, m.G.forward(m.G.w, Z; mode=false))
	G_sample .-= minimum(G_sample; dims =(1,2))
	G_sample ./= maximum(G_sample; dims =(1,2))
	out = map(clamp01nan, Gray.(makegrid(G_sample; gridsize=(gs,gs), scale=scale, shape=(s,s))))

	savefile = string(printfolder, "/epoch",lpad(epoch, 3, "0"),".png")
	save(savefile, out)
end
export print_output


function train(xtrn::Union{KnetArray, Array}, ytrn::Union{KnetArray, Array},
			xtst::Union{KnetArray, Array}, ytst::Union{KnetArray, Array},
			model::InfoModel; mdlfile="../trained/model.jld2", logfile="../logs/log.txt", printfolder="outputs")
	epochs, bs = model.o[:epochs], model.o[:batchsize]
	traindata = Array{Float64}(undef, epochs, 2)

	bestVal = 0
	for epoch = 1:epochs
		fh = open(logfile, "a")
		epoch%2>0 && model.o[:v] ? verbose=true : verbose=false

		trnIdx = mbatch(randperm(size(xtrn,4)), bs)
		lossD, lossG = 0, 0
		total = length(trnIdx)
		for i = 1:total
			batchv = verbose && i%20==0

			y = ytrn[:, trnIdx[i]]
			x_orig = xtrn[:, :, :, trnIdx[i]]
			#train D
			for j=1:model.o[:dreps]
				x = convert(model.F.atype, augment(x_orig))
				batch_lossD, xs = D_loss(model, x; mode=true, v=(batchv && j==model.o[:dreps]))
				trainD!(model, xs)
				lossD += batch_lossD
		    end

		    #train G
		    for k=1:model.o[:greps]
				x = convert(model.F.atype, augment(x_orig))
				batch_lossG, Zinfo = G_loss(model, x, y; mode=true, v=(batchv && k==model.o[:greps]))
				trainG!(model, Zinfo, x, y)
				lossG += batch_lossG
		    end
		end

		# Check Validation Accuracy
		val_acc = validation_accuracy(xtst, ytst, model)
		verbose && @printf "Epoch Classification Accuracy: %.3f \t(best %.3f)\n" val_acc bestVal
		write(fh, @sprintf "Epoch Classification Accuracy: %.3f \t(best %.3f)\n" val_acc bestVal)
		#
		lossG /= total; lossD /= total;
		verbose && @printf("epoch: %d loss[D]: %.3e loss[G]: %.3e\n", epoch, lossD, lossG)
		write(fh, "epoch $epoch \tloss[D] \t$lossD \tloss[G] \t$lossG\n")
		traindata[epoch, 1:2] = [lossD, lossG]

		(model.o[:printNum] > 0) && print_output(epoch, model, size(xtrn,1), 2.0, 10; printfolder=printfolder)
		(val_acc > bestVal) && (bestVal=val_acc; save_model(model; savefile=mdlfile))
		close(fh)
	end
	return traindata
end
export train

function validation_accuracy(xtst::Union{KnetArray, Array}, ytst::Union{KnetArray, Array}, model::InfoModel)
	bs = model.o[:batchsize]
	N = size(xtst,4)
	tstIdx = mbatch(Array(1:N), bs)
	total = length(tstIdx)

	all_logits = [[] for i in 1:length(model.o[:c_SS])]
	for batch = 1:total
		i = bs * (batch-1) + 1
		j = min(N, i + bs - 1)

		x = convert(model.F.atype, xtst[:, :, :, tstIdx[batch]])
		fc = model.F.forward(model.F.w, x; mode=false)
		logitsSS, _, _, _ = model.Q.forward(model.Q.w, fc, model.o; mode=false)
		for k = 1:length(model.o[:c_SS])
			push!(all_logits[k], Array(logitsSS[k]))
		end
	end

	total_acc = 0
	for k = 1:length(model.o[:c_SS])
		scores = hcat(all_logits[k]...)
		answers = Array{Int}(ytst[k:k,:])
		total_acc += accuracy(scores, answers; dims=1)
	end
	total_acc /= length(model.o[:c_SS])
	return total_acc
end

function get_c(xtst::Union{KnetArray, Array}, model::InfoModel)
	bs = model.o[:batchsize]
	N = size(xtst,4)

	tstIdx = mbatch(Array(1:N), bs)
	total = length(tstIdx)

	c = Array{Int64}(undef, length(model.o[:c_SS]), N)

	for batch = 1:total
		i = bs * (batch-1) + 1
		j = min(N, i + bs - 1)

		x = convert(model.F.atype, xtst[:, :, :, tstIdx[batch]])
		fc = model.F.forward(model.F.w, x; mode=false)
		logitsSS, _, _, _ = model.Q.forward(model.Q.w, fc, model.o; mode=false)
		for k = 1:length(logitsSS)
			c[k, i:j] = map(cc->cc[1], argmax(Array(logitsSS[k]); dims=1))
		end
	end
	return c
end
get_c(xtst::Union{KnetArray, Array}) = get_c(xtst, "../trained/model.jld2")
get_c(xtst::Union{KnetArray, Array}, mdlfile::String) = (model = load_model(mdlfile); get_c(xtst, model))

export get_c


function mbatch(idx, batchsize)
    N = length(idx)
    nbatches = Int(ceil(N/batchsize))
    data = Array{Array{Int64,1}}(undef, nbatches)
    for m =1:nbatches
        i = batchsize*(m-1) + 1
        j = min(N, i + batchsize - 1)
        data[m] = idx[i:j]
    end
    return data
end
export mbatch

function augment(data_orig)
    data = copy(data_orig)
    bs = size(data, 4)

    flipIDs = rand([0,0,1,2], bs)
    for i in 1:bs
        (flipIDs[i] > 0) && (data[:,:,:,i] = reverse(data[:,:,:,i], dims=flipIDs[i]))
    end

    rotIDs = rand([0,1,2,3], bs)
    for j in 1:bs
        (flipIDs[j] > 0) && (data[:,:,:,j] = rotate_im(data[:,:,:,j], rotIDs[j]))
    end
    return data
end
export augment

function rotate_im(im, n)
    for ch in 1:size(im,3)
        if n == 1
            im[:,:,ch] = rotr90(im[:,:,ch])
        elseif n == 2
            im[:,:,ch] = rot180(im[:,:,ch])
        elseif n == 3
            im[:,:,ch] = rotl90(im[:,:,ch])
        end
    end
    return im
end

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
        k % gridy == 0 ? (x0 = x1+2; y0 = 2;) : y0 = y1+2
    end
    return convert(Array{Float64,2}, map(x->isnan(x) ? 0 : x, out))
end

save_results(filename, results) = save(filename,  Dict("results" => results))
export save_results

end
