# infoganJL
InfoGAN implementation in Julia with flexibility generator, descriminator, and auxiliary neural network architectures. InfoGAN model structure is implemented as described in [https://arxiv.org/abs/1606.03657] with semi-supervised training capacity for auxiliary networks. Additionally, the Discriminator output and loss are designed according to Wassterstein GAN innovations [https://arxiv.org/abs/1701.07875]"

Run trianing with default settings by running `julia infoGAN.jl`. This will load our experimental digital holography dataset from the resources file, and implement default neural network architectures and learning parameters. Parameters can be customized by providing additional arguments in the initial call.

Modifications can be made to InfoGAN.loaddata() located in helpers.jl to deal with new datasets. Default neural network architectures are also stored in helpers.jl and can be modifiied for new tasks.

If running InfoGAN module in a notebook environment, neural network archtectures can be defined prior to creating the initial InfoModel structure. Once an InfoModel strucuture is defined from hyperparameters in `o` and from neural network weights/functions, we can run the `train(xtrn, ytrn, xtst, ytst, model; mdlfile=MDLFILE, logfile=LOGFILE, printfolder=PRINTFOLDER)` function to start training. A demo is shown in demo.ipynb.
