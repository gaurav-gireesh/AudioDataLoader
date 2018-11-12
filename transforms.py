# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# coding: utf-8
# pylint: disable= arguments-differ
"Audio transforms."

import numpy as np
import librosa
import mxnet as mx
from mxnet import nd
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

        audio_tmp = np.mean(librosa.feature.mfcc(y=x.asnumpy(), sr=22050, n_mfcc=40).T, axis=0)
        return nd.array(audio_tmp)


class Scale(Block):
    """Scale audio numpy.ndarray from a 16-bit integer to a floating point number between
    -1.0 and 1.0. The 16-bit integer is the sample resolution or bit depth.

    Args:
        factor (int): maximum value of input tensor. default: 16-bit depth

    Examples
    --------
    >>> scale = audio.transforms.Scale(scale_factor=2)
    >>> audio_samples = mx.nd.array([2,3,4])
    >>> scale(audio_samples)
    [1.  1.5 2. ]
    <NDArray 3 @cpu(0)>

    """

    def __init__(self, scale_factor=2**31, **kwargs):
        self.scale_factor = scale_factor
        super(Scale, self).__init__(**kwargs)

    def forward(self, x):
        """
        Args:
            x : NDArray of audio of size (Number of samples X Number of channels(1 for mono, >2 for stereo))

        Returns:
            NDArray: Scaled by the scaling factor. (default between -1.0 and 1.0)

        """
        return x / self.scale_factor


class PadTrim(Block):
    """Pad/Trim a 1d-NDArray of NPArray (Signal or Labels)

    Args:
        x (NDArray): Array( numpy.ndarray or mx.nd.NDArray) of audio of shape (samples, )
        max_len (int): Length to which the array will be padded or trimmed to.
        fill_value: If there is a need of padding, what value to padd at the end of the input x

    Examples
    --------
    >>> padtrim = audio.transforms.PadTrim(max_len=9, fill_value=0)
    >>> audio_samples = mx.nd.array([1,2,3,4,5])
    >>> padtrim(audio_samples)
    [1. 2. 3. 4. 5. 0. 0. 0. 0.]
    <NDArray 9 @cpu(0)>

    """

    def __init__(self, max_len, fill_value=0, **kwargs):
        self._max_len = max_len
        self._fill_value = fill_value
        super(PadTrim, self).__init__(**kwargs)

    def forward(self, x):
        """

        Returns:
            Tensor: (1 x max_len)

        """
        if  isinstance(x, np.ndarray):
            x = mx.nd.array(x)
        if self._max_len > x.size:
            pad = mx.nd.ones((self._max_len - x.size,)) * self._fill_value
            x = mx.nd.concat(x, pad, dim=0)
        elif self._max_len < x.size:
            x = x[:self._max_len]
        return x
    