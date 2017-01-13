#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2016-2017 CNRS

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# AUTHORS
# Hervé BREDIN - http://herve.niderb.fr


from ._version import get_versions
__version__ = get_versions()['version']
del get_versions


from pyannote.database import Database
from pyannote.database.protocol import SpeakerDiarizationProtocol
from pyannote.parser import UEMParser
from pyannote.parser import MDTMParser
import os.path as op


class OdessaAMISpeakerDiarizationProtocol(SpeakerDiarizationProtocol):
    """Base speaker diarization protocol for ODESSA/AMI database

    This class should be inherited from, not used directly.

    Parameters
    ----------
    preprocessors : dict or (key, preprocessor) iterable
        When provided, each protocol item (dictionary) are preprocessed, such
        that item[key] = preprocessor(**item). In case 'preprocessor' is not
        callable, it should be a string containing placeholder for item keys
        (e.g. {'wav': '/path/to/{uri}.wav'})
    """

    def __init__(self, preprocessors={}, **kwargs):
        super(OdessaAMISpeakerDiarizationProtocol, self).__init__(
            preprocessors=preprocessors, **kwargs)
        self.mdtm_parser_ = MDTMParser()
        self.uem_parser_ = UEMParser()

    def _subset(self, protocol, subset):

        data_dir = op.join(op.dirname(op.realpath(__file__)), 'data')

        # load uems
        path = op.join(data_dir, '{protocol}.{subset}.uem'.format(subset=subset, protocol=protocol))
        uems = self.uem_parser_.read(path)

        # load annotations
        path = op.join(data_dir, '{protocol}.{subset}.mdtm'.format(subset=subset, protocol=protocol))
        mdtms = self.mdtm_parser_.read(path)

        for uri in sorted(mdtms.uris):
            annotation = mdtms(uri)
            annotated = uems(uri)
            current_file = {
                'database': 'AMI',
                'uri': uri,
                'annotation': annotation,
                'annotated': annotated}
            yield current_file


class P1(OdessaAMISpeakerDiarizationProtocol):
    """ODESSA/AMI P1 protocol

    Parameters
    ----------
    preprocessors : dict or (key, preprocessor) iterable
        When provided, each protocol item (dictionary) are preprocessed, such
        that item[key] = preprocessor(**item). In case 'preprocessor' is not
        callable, it should be a string containing placeholder for item keys
        (e.g. {'wav': '/path/to/{uri}.wav'})
    """

    def trn_iter(self):
        return self._subset('p1', 'trn')

    def dev_iter(self):
        return self._subset('p1', 'dev')

    def tst_iter(self):
        return self._subset('p1', 'tst')


class P2(OdessaAMISpeakerDiarizationProtocol):
    """ODESSA/AMI P2 protocol

    Parameters
    ----------
    preprocessors : dict or (key, preprocessor) iterable
        When provided, each protocol item (dictionary) are preprocessed, such
        that item[key] = preprocessor(**item). In case 'preprocessor' is not
        callable, it should be a string containing placeholder for item keys
        (e.g. {'wav': '/path/to/{uri}.wav'})
    """

    def trn_iter(self):
        return self._subset('p2', 'trn')

    def dev_iter(self):
        return self._subset('p2', 'dev')

    def tst_iter(self):
        return self._subset('p2', 'tst')


class AMI(Database):
    """AMI corpus

Parameters
----------
preprocessors : dict or (key, preprocessor) iterable
    When provided, each protocol item (dictionary) are preprocessed, such
    that item[key] = preprocessor(**item). In case 'preprocessor' is not
    callable, it should be a string containing placeholder for item keys
    (e.g. {'wav': '/path/to/{uri}.wav'})

Reference
---------

Citation
--------

Website
-------
    """

    def __init__(self, preprocessors={}, **kwargs):
        super(AMI, self).__init__(preprocessors=preprocessors, **kwargs)

        self.register_protocol(
            'SpeakerDiarization', 'P1', P1)

        self.register_protocol(
            'SpeakerDiarization', 'P2', P2)
