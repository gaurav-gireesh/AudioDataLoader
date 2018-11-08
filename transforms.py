import librosa
import mxnet as mx
from mxnet import gluon, nd
import numpy as np
from mxnet.gluon.block import Block


class Loader(Block):
    """
        This transform opens a filepath and converts that into an NDArray using librosa to load
    """
    def __init__(self, **kwargs):
        super(Loader, self).__init__(**kwargs)
        
    def forward(self, x):
        if not librosa:
            raise RuntimeError("Librosa dependency is not installed! Install that and retry!")
        X1, _ = librosa.load(x, res_type='kaiser_fast')

        return nd.array(X1)


class MFCC(Block):
    """
        Extracts Mel frequency cepstrum coefficients from the audio data file
        More details : https://librosa.github.io/librosa/generated/librosa.feature.mfcc.html

        returns:    An NDArray after extracting mfcc features from the input
    """
    def __init__(self, **kwargs):
        super(MFCC, self).__init__(**kwargs)
        
    def forward(self, x):
        if not librosa:
            raise RuntimeError("Librosa dependency is not installed! Install that and retry")
        
        audio_tmp = np.mean(librosa.feature.mfcc(y=x.asnumpy(), sr=22050, n_mfcc=40).T,axis=0)
        return nd.array(audio_tmp)