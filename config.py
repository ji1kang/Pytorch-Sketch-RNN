# hyperparameters
class HParams():
    def __init__(self):
        self.data_dir = '/media/scteam2/quickdraw_dataset/npz'
        self.epoch = 1000
        self.data_set = ['flower.npz', 'umbrella.npz', 'tornado.npz', 'pool.npz',
                         'tree.npz', 'rain.npz', 'rainbow.npz', 'cloud.npz', 'face.npz']  # Our dataset.
        # Not used. Will be changed by model. [Eliminate?]
        self.max_seq_len = 250

        '''
        #self.data_set = ['flower.npz', 'face.npz']
        self.output_dim = len(self.data_set)
        self.enc_hidden_size = 256
        self.dec_hidden_size = 512
        self.z_size = 128 # Size of latent vector z. Recommend 32, 64 or 128.
        self.M = 20 # Number of mixtures in Gaussian mixture model.
        self.batch_size=512  # Minibatch size. Recommend leaving at 100.
        self.eta_min = 0.01 # KL start weight when annealing.
        self.R = 0.99995 # KL annealing decay rate per minibatch.
        self.KL_min = 0.2 # Level of KL loss at which to stop optimizing for KL.
        self.wKL = 0.5 # KL weight of loss equation. Recommend 0.5 or 1.0.
        self.lr = 0.001 # Learning rate.
        self.lr_decay = 0.9999  # Learning rate decay per minibatch.
        self.min_lr = 0.00001 # Minimum learning rate.
        self.grad_clip = 1.0# Gradient clipping. Recommend leaving at 1.0.
        self.temperature = 0.5
        
        self.max_seq_len = 250 # Not used. Will be changed by model. [Eliminate?]
        self.use_input_dropout = False  # Input dropout. Recommend leaving False.
        self.input_dropout_prob=0.90  # Probability of input dropout keep.
        self.use_recurrent_dropout = True  # Dropout with memory loss. Recomended
        self.recurrent_dropout_prob=0.90  # Probability of recurrent dropout keep
        self.is_training = True  # Is model training? Recommend keeping true.
        self.random_scale_factor = 0.15  # Random scaling data augmention proportion.
        self.augment_stroke_prob = .10  # Point dropping augmentation proportion.
        self.normalizing_scale_factor = 1.000003 # trainset_normalizing_scale_factor
        self.checkpoint = '/home/dsail/jiwon/mental-health/draw/models/sketchrnn_pytorch/'
        '''
