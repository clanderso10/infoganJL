push!(LOAD_PATH, pwd())

using Pkg
for p = ["Knet", "Printf", "ArgParse", "Compat",
	"GZip", "Images", "JLD2", "SparseArrays",
	"Distributions", "Random", "MAT", "ImageMagick"]
	haskey(Pkg.installed(), p) || Pkg.add(p)
end

module InfoGAN
using InfoGAN_Fns, helpers
using Knet, ArgParse, Printf, Random, JLD2, Images, Dates
using Dates: format

function define_params(args)
	s = ArgParseSettings(prog="InfoGAN.jl")
	s.description = "Implementation of InfoGAN model described in [https://arxiv.org/abs/1606.03657] with semi-supervised
	training capacity for auxiliary networks, and updated Discriminator loss function according to [https://arxiv.org/abs/1701.07875]"
	s.exc_handler=ArgParse.debug_handler

	add_arg_group(s, "Training Hyperparameters")
	@add_arg_table s begin
		("--epochs"; arg_type=Int; default=100; help= "Number of times to iterate through dataset")
		("--batchsize"; arg_type=Int; default=64; help= "Size of data minibatches")
		("--lr"; arg_type=Float64; nargs=4; default=[1e-5, 1e-5, 1e-4, 1e-4]; help = "Learning rates for AdamGrad optimizers (Front End, Discriminator, Generator, Auxiliary)")
		("--clip"; arg_type=Float64; default=.5 ; help= "WGAN weight-clipping coefficient")
		("--dreps"; arg_type=Int; default=5; help="Number of Discriminator-side training repeats per minibatch")
		("--greps"; arg_type=Int; default=1; help="Number of Generator-side training repeats per minibatch")
		("--mdl"; default="trained/model.jld2"; help="File where trained model is saved/should be saved")
		("--load"; action = :store_true; help="Load previous model before training")
	end

	add_arg_group(s, "Latent Code Hyperparameters")
	@add_arg_table s begin
		("--z"; arg_type=Int; default=74; help="Size of noise component, z")
		("--c_SS"; arg_type=Int; nargs='+'; default=[2, 2] ; help="Size of each semisupervised discrete code (array)")
		("--c_disc"; arg_type=Int; nargs='*' ; help="Size of each unsupervised discrete code (array)")
		("--c_cont"; arg_type=Int; default=0 ; help="Number of real-valued codes (int)")
		("--discCoeffSS"; arg_type=Float64; default=1.0 ; help="Loss Coefficient for semisupervised Discrete Variables")
		("--discCoeff"; arg_type=Float64; default=1.0 ; help="Loss Coefficient for unsupervised Discrete Variables")
	end

	add_arg_group(s, "Data Set Hyperparameters")
	@add_arg_table s begin
		("--datadir"; arg_type=String; default="../resources/training_data"; help="Directory location holding holography dataset. Use 'MNIST' to load MNIST dataset instead")
		("--N"; arg_type=Int; default=nothing ; help="Number of dataset samples to use for training")
		("--atype"; eval_arg=true; default=(gpu()>=0 ? "KnetArray{Float32}" : "Array{Float32}"); help="Array type for training data: Array for cpu, KnetArray for gpu")
	end

	add_arg_group(s, "Other parameters")
	@add_arg_table s begin
		("--printNum"; arg_type=Int; default=100; help="Number of images that generator function produces (default folder ../outputs/). Set to 0 for no printing")
		("-v"; action = :store_true; help = "verbose flag, use to print updates every 2 epochs")
	end


	isa(args, AbstractString) && (args=split(args));
	if in("--help", args) || in("-h", args)
		ArgParse.show_help(s; exit_when_done=false)
		return nothing
	end

	o = (length(args) > 0 ? parse_args(args, s; as_symbols=true) : parse_args(s; as_symbols=true))
	o[:atype] = eval(Meta.parse(o[:atype]))
	return o
end
export define_params


function main(args)
	# GeneratorLayers = default_generator()

	o = define_params(args); o==nothing && return;
	@info "GAN Started..." o

	# Load dataset
	(xtrn, xtst, ytrn, ytst) = loaddata(o)
	if o[:N] != nothing
		trnN = size(xtrn,4)
		selection = randperm(trnN)[1:min(trnN, o[:N])]
		xtrn = xtrn[:,:,:,selection]
		ytrn = ytrn[:, selection]
	end
	@info "Size of Datasets: " size(xtrn) size(xtst) size(ytrn) size(ytst)

	if(ispath(o[:mdl]) && o[:load])
		model = load_model(o[:mdl])
	else
		@info "No weights found, initializing new ones"
		model = InfoModel(o)
	end
	MDLFILE = model.o[:mdl]

	LOGFILE = @sprintf "logs/log-%s.txt" format(now(), "mmddyy-HH")
	fh = open(LOGFILE, "w")
	for k=keys(o); write(fh, string(k, "\t\t\t", o[k], "\n")); end
	close(fh)

	@time results = train(xtrn, ytrn, xtst, ytst, model; logfile=LOGFILE, mdlfile=MDLFILE)
	c = get_c(xtst, MDLFILE)
	save("trained/test_categories.jld2", Dict("preds"=> c, "labels"=> ytst))

	RESFILE = @sprintf "trained/results-%s.jld2" format(now(), "mmddyy-HH")
	save_results(RESFILE,results)
end
export main

PROGRAM_FILE=="InfoGAN.jl" && main(ARGS);
end
