# infoganJL
InfoGAN implementation in Julia with flexibility generator, descriminator, and auxiliary neural network architectures. InfoGAN model structure is implemented as described in [https://arxiv.org/abs/1606.03657] with semi-supervised training capacity for auxiliary networks. Additionally, the Discriminator output and loss are designed according to Wassterstein GAN innovations [https://arxiv.org/abs/1701.07875]"

Run trianing with default settings by running `julia infoGAN.jl`. This will load our experimental digital holography dataset from the resources file, and implement default neural network architectures and learning parameters. Parameters can be customized by providing additional arguments in the initial call.

Training Hyperparameters:
  --epochs EPOCHS         Number of times to iterate through dataset
  --batchsize BATCHSIZE   Size of data minibatches 
  --lr LR LR LR LR        Learning rates for AdamGrad optimizers (Front
                          End, Discriminator, Generator, Auxiliary)
  --clip CLIP             WGAN weight-clipping coefficient
  --dreps DREPS           Number of Discriminator-side training repeats
                          per minibatch
  --greps GREPS           Number of Generator-side training repeats per
                          minibatch
  --mdl MDL               File where trained model is saved/should be
                          saved
  --load                  Load previous model before training

Latent Code Hyperparameters:
  --z Z                   Size of noise component, z
  --c_SS C_SS [C_SS...]   Size of each semisupervised discrete code
                          (array)
  --c_disc [C_DISC...]    Size of each unsupervised discrete code
                          (array)
  --c_cont C_CONT         Number of real-valued codes (int)
  --discCoeffSS DISCCOEFFSS Loss Coefficient for semisupervised Discrete
                            Variables
  --discCoeff DISCCOEFF   Loss Coefficient for unsupervised Discrete
                          Variables

Data Set Hyperparameters:
  --datadir DATADIR       Directory location holding holography
                          dataset. Use 'MNIST' to load MNIST dataset
                          instead 
  --N N                   Number of dataset samples to use for training
  --atype ATYPE           Array type for training data: Array for cpu,
                          KnetArray for gpu

Other parameters:
  --printNum PRINTNUM   Number of images that generator function
                        produces (default folder ./outputs/). Set to 0
                        for no printing
  -v                    verbose flag, use to print updates every 2
                        epochs

Modifications can be made to InfoGAN.loaddata() located in helpers.jl to deal with new datasets. Default neural network architectures are also stored in helpers.jl and can be modifiied for new tasks.

If running InfoGAN module in a notebook environment, neural network archtectures can be defined prior to creating the initial InfoModel structure. Once an InfoModel strucuture is defined from hyperparameters in `o` and from neural network weights/functions, we can run the `train(xtrn, ytrn, xtst, ytst, model; mdlfile=MDLFILE, logfile=LOGFILE, printfolder=PRINTFOLDER)` function to start training. A demo is shown in demo.ipynb.
