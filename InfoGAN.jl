push!(LOAD_PATH, pwd())

using Pkg
for p = ["Knet", "Printf", "ArgParse", "Compat",
	"GZip", "Images", "JLD2", "SparseArrays",
	"Distributions", "Random", "MAT", "ImageMagick"]
	haskey(Pkg.installed(), p) || Pkg.add(p)
end

module InfoGAN
using InfoGAN_Fns, helpers
using Knet, ArgParse, Printf, Random

function define_params(args)
	s = ArgParseSettings()
	s.description = "Implementation of InfoGAN model described in [https://arxiv.org/abs/1606.03657] with semi-supervised training capacity for auxiliary network";
	s.exc_handler=ArgParse.debug_handler

	@add_arg_table s begin
	  ("--N"; arg_type=Int; default=15448 ; help="Number of dataset samples to use")
	  ("--epochs"; arg_type=Int; default=100; help="Number of times to iterate through dataset")
	  ("--batchsize"; arg_type=Int; default=64; help="Size of minibatch updates")
	  ("--lr"; arg_type=Union{Int, NTuple{4, Float64}}; default=(1e-6, 5e-5, 5e-4, 5e-4); help = "Learning rates for AdamGrad optimizers (Front End, Discriminator, Generator, Auxiliary)")
	  ("--dreps"; arg_type=Int; default=3; help="Number of Discriminator-side training repeats per minibatch")
	  ("--greps"; arg_type=Int; default=1; help="Number of Generator-side training repeats per minibatch")

	  ("--z"; arg_type=Int; default=74; help="Size of noise component, z")
	  ("--c_disc"; arg_type=Union{Tuple, Int}; default=(2,2) ; help="Sizes of all discrete codes (tuple)")
	  ("--c_cont"; arg_type=Int; default=1 ; help="Number of continuous codes (int)")
	  ("--atype"; default=(gpu()>=0 ? "KnetArray{Float32}" : "Array{Float32}"); help="array type: Array for cpu, KnetArray for gpu")

	  ("--print"; default=true ; help="Set false to turn off creating output images")
	  ("--gencnt"; arg_type=Int; default=100; help="Number of images that generator function creates.")
	  ("-v"; action = :store_true; help = "verbose flag, use to print updates every 2 epochs")
	end

	isa(args, AbstractString) && (args=split(args));
	if in("--help", args) || in("-h", args)
		ArgParse.show_help(s; exit_when_done=false)
		return nothing
	end

	println(s.description)
	if length(args) > 0
		o = parse_args(args, s; as_symbols=true)
	else
		o = parse_args(s; as_symbols=true)
	end
	o[:atype] = eval(Meta.parse(o[:atype]))
	return o
end
export define_params


function main(args)
	o = define_params(args); o==nothing && return;
	@info "GAN Started..." o

	#Load dataset
	datadir = string(pwd(), "/training_data")
	(xtrn,xtst,ytrn,ytst) = loaddata(Array{Float32}; dataset=datadir, N=o[:N])
	@info "Size of Datasets: " size(xtrn) size(xtst) size(ytrn) size(ytst)
	#tst = minibatch(xtst, ytst, o[:batchsize];atype=atype)
	# size(xtrn)[end] == size(ytrn)[end] || throw(DimensionMismatch())

	if(ispath("Dnet.jld") && ispath("Gnet.jld"))
		# Gnet = load("Gnet.jld","Gnet")
	else
		@info "No weights found, initializing new ones"
		model = InfoModel(o)
	end
	@time train(xtrn, ytrn, model, o; logfile="log_default.txt")
end
export main

PROGRAM_FILE=="InfoGAN.jl" && main(ARGS);

end
