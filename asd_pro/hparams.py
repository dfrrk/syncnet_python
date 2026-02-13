class HParams:
    def __init__(self, **kwargs):
        self.sample_rate = 16000
        self.n_fft = 800
        self.hop_size = 200
        self.win_size = 800
        self.num_mels = 80
        self.fmin = 55
        self.fmax = 7600
        self.ref_level_db = 20
        self.min_level_db = -100
        self.preemphasis = 0.97
        self.preemphasize = True
        self.signal_normalization = True
        self.allow_clipping_in_normalization = True
        self.symmetric_mels = True
        self.max_abs_value = 4.
        self.use_lws = False
        self.__dict__.update(kwargs)

hparams = HParams()
