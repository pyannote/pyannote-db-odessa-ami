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


from pyannote.core import Segment, Timeline, Annotation
from pyannote.database import Database
from pyannote.database.protocol import SpeakerDiarizationProtocol
from pyannote.database.protocol import SpeakerSpottingProtocol
from pyannote.parser import UEMParser
from pyannote.parser import MDTMParser
from pandas import read_table
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


class P1MH(OdessaAMISpeakerDiarizationProtocol):
    """ODESSA/AMI P1MH (mix-headset) protocol

    Parameters
    ----------
    preprocessors : dict or (key, preprocessor) iterable
        When provided, each protocol item (dictionary) are preprocessed, such
        that item[key] = preprocessor(**item). In case 'preprocessor' is not
        callable, it should be a string containing placeholder for item keys
        (e.g. {'wav': '/path/to/{uri}.wav'})
    """

    def trn_iter(self):
        return self._subset('p1mh', 'trn')

    def dev_iter(self):
        return self._subset('p1mh', 'dev')

    def tst_iter(self):
        return self._subset('p1mh', 'tst')


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


class P2MH(OdessaAMISpeakerDiarizationProtocol):
    """ODESSA/AMI P2MH (mix-headset) protocol

    Parameters
    ----------
    preprocessors : dict or (key, preprocessor) iterable
        When provided, each protocol item (dictionary) are preprocessed, such
        that item[key] = preprocessor(**item). In case 'preprocessor' is not
        callable, it should be a string containing placeholder for item keys
        (e.g. {'wav': '/path/to/{uri}.wav'})
    """

    def trn_iter(self):
        return self._subset('p2mh', 'trn')

    def dev_iter(self):
        return self._subset('p2mh', 'dev')

    def tst_iter(self):
        return self._subset('p2mh', 'tst')


class MixHeadsetSpeakerSpotting(SpeakerSpottingProtocol):

    def _xxx_iter(self, subset):

        data_dir = op.join(op.dirname(op.realpath(__file__)), 'data')

        # reference
        ref_template = 'AMI.split_references/AMI.p1mh.splitSessionsWithOffset.{subset}.rttm'
        ref_path = op.join(data_dir, 'llss',
                           ref_template.format(subset=subset))
        ref = read_table(
            ref_path, delim_whitespace=True,
            names=['SPEAKER', 'session_id', '1', 'start', 'duration',
                   'NA1', 'NA2', 'speaker_id', 'NA3', 'NA4'])
        ref = ref.groupby('session_id')

        # session mapping
        session_template = 'AMI.split_references/AMI.p1mh.splitSessionsMapping.{subset}.lst'
        session_path = op.join(data_dir, 'llss',
                           session_template.format(subset=subset))
        sessions = read_table(
            session_path, delim_whitespace=True,
            names=['session_id', 'uri', 'start', 'end'])

        for session in sessions.itertuples():

            uri = session.uri + '.Mix-Headset'
            crop_uri = session.session_id

            session_id = session.session_id

            annotation = Annotation(uri=uri)
            crop_annotation = Annotation(uri=crop_uri)

            for i, turn in enumerate(ref.get_group(session_id).itertuples()):

                segment = Segment(turn.start,
                                  turn.start + turn.duration)

                crop_segment = Segment(
                    turn.start - session.start,
                    turn.start - session.start + turn.duration)

                speaker_id = turn.speaker_id

                annotation[segment, i] = speaker_id
                crop_annotation[crop_segment, i] = speaker_id

            annotated = Timeline(
                segments=[Segment(session.start, session.end)],
                uri=uri)

            crop_annotated = Timeline(
                segments=[Segment(0., session.end - session.start)],
                uri=crop_uri)

            current_file = {
                'uri': uri,
                'database': 'AMI',
                'annotation': annotation,
                'annotated': annotated,
                'crop': {
                    'uri': crop_uri,
                    'database': 'AMI',
                    'annotation': crop_annotation,
                    'annotated': crop_annotated
                }
            }

            yield current_file

    def trn_iter(self):
        return self._xxx_iter('trn')

    def dev_iter(self):
        return self._xxx_iter('dev')

    def tst_iter(self):
        return self._xxx_iter('tst')

    def _xxx_enrol_iter(self, subset):

        data_dir = op.join(op.dirname(op.realpath(__file__)), 'data')

        # reference
        ref_template = 'AMI.p1mh/{subset}/AMI.p1mh.enrollment_60sec.enrollment.{subset}.rttm'
        ref_path = op.join(data_dir, 'llss',
                           ref_template.format(subset=subset))
        ref = read_table(
            ref_path, delim_whitespace=True,
            names=['SPEAKER', 'session_id', '1', 'start', 'duration',
                   'NA1', 'NA2', 'model_id', 'NA3', 'NA4'])
        ref = ref.groupby(['session_id', 'model_id'])

        # session mapping
        session_template = 'AMI.split_references/AMI.p1mh.splitSessionsMapping.{subset}.lst'
        session_path = op.join(data_dir, 'llss',
                           session_template.format(subset=subset))
        sessions = read_table(
            session_path, delim_whitespace=True,
            names=['session_id', 'uri', 'start', 'end'],
            index_col='session_id')

        # models
        model_template = 'AMI.p1mh/{subset}/AMI.p1mh.enrollment_60sec.speakerModels.{subset}.lst'
        model_path = op.join(data_dir, 'llss',
                             model_template.format(subset=subset))
        models = read_table(
            model_path, delim_whitespace=True,
            names=['model_id', 'session_id'])

        for model in models.itertuples():

            session_id = model.session_id
            session = sessions.loc[session_id]

            uri = session.uri + '.Mix-Headset'
            crop_uri = session_id

            model_id = model.model_id

            try:
                speech_turns = ref.get_group((session_id, model_id))
            except KeyError as e:
                print('Failure', model_id)
                continue

            segments = []
            crop_segments = []
            for turn in speech_turns.itertuples():

                if turn.duration == 0.:
                    continue

                segment = Segment(
                    turn.start,
                    turn.start + turn.duration)

                crop_segment = Segment(
                    turn.start - session.start,
                    turn.start - session.start + turn.duration)

                segments.append(segment)
                crop_segments.append(crop_segment)

            enrol_with = Timeline(segments=segments, uri=uri)
            crop_enrol_with = Timeline(segments=crop_segments, uri=crop_uri)

            current_enrolment = {
                'database': 'AMI',
                'uri': uri,
                'model_id': model_id,
                'enrol_with': enrol_with,
                'crop': {
                    'database': 'AMI',
                    'uri': crop_uri,
                    'model_id': model_id,
                    'enrol_with': crop_enrol_with
                }
            }

            yield current_enrolment

    def trn_enrol_iter(self):
        return self._xxx_enrol_iter('trn')

    def dev_enrol_iter(self):
        return self._xxx_enrol_iter('dev')

    def tst_enrol_iter(self):
        return self._xxx_enrol_iter('tst')

    def _xxx_try_iter(self, subset):

        data_dir = op.join(op.dirname(op.realpath(__file__)), 'data')

        # reference
        ref_template = 'AMI.split_references/AMI.p1mh.splitSessionsWithOffset.{subset}.rttm'
        ref_path = op.join(data_dir, 'llss',
                           ref_template.format(subset=subset))
        ref = read_table(
            ref_path, delim_whitespace=True,
            names=['SPEAKER', 'session_id', '1', 'start', 'duration',
                   'NA1', 'NA2', 'speaker_id', 'NA3', 'NA4'])
        ref = ref.groupby(['session_id', 'speaker_id'])

        # list of trials
        trl_template = 'AMI.p1mh/{subset}/AMI.p1mh.enrollment_60sec.LLSS.{subset}.trl'
        trl_path = op.join(data_dir, 'llss',
                           trl_template.format(subset=subset))
        trials = read_table(
            trl_path, delim_whitespace=True,
            names=['model_id', 'session_id', 'start', 'trial'])

        # session mapping
        session_template = 'AMI.split_references/AMI.p1mh.splitSessionsMapping.{subset}.lst'
        session_path = op.join(data_dir, 'llss',
                           session_template.format(subset=subset))
        sessions = read_table(
            session_path, delim_whitespace=True,
            names=['session_id', 'uri', 'start', 'end'],
            index_col='session_id')

        for trial in trials.itertuples():

            model_id = trial.model_id
            session_id = trial.session_id

            try:
                session = sessions.loc[session_id]
            except Exception as e:
                print('Failure session {session_id}'.format(session_id=session_id))
                continue

            uri = session.uri + '.Mix-Headset'
            crop_uri = session_id

            try_with = Segment(session.start, session.end)
            crop_try_with = Segment(0, session.end - session.start)

            # TODO / check if this is always true...
            # otherwise we may need another mapping file
            speaker_id = model_id.split('_')[0]

            segments = []
            crop_segments = []
            try:
                speech_turns = ref.get_group((session_id, speaker_id))
                for turn in speech_turns.itertuples():

                    if turn.duration < 0:
                        print('Negative duration {session_id} / {speaker_id}'.format(session_id=session_id, speaker_id=speaker_id))
                        continue

                    if turn.duration == 0.:
                        continue

                    segment = Segment(
                        turn.start,
                        turn.start + turn.duration)

                    crop_segment = Segment(
                        turn.start - session.start,
                        turn.start - session.start + turn.duration)

                    segments.append(segment)
                    crop_segments.append(crop_segment)

            except KeyError as e:
                pass

            reference = Timeline(segments=segments, uri=uri)
            crop_reference = Timeline(segments=crop_segments, uri=crop_uri)

            current_trial = {
                'database': 'AMI',
                'uri': uri,
                'try_with': try_with,
                'model_id': model_id,
                'reference': reference,
                'crop': {
                    'database': 'AMI',
                    'uri': crop_uri,
                    'try_with': crop_try_with,
                    'model_id': model_id,
                    'reference': crop_reference
                }
            }

            yield current_trial

    def trn_try_iter(self):
        return self._xxx_try_iter('trn')

    def dev_try_iter(self):
        return self._xxx_try_iter('dev')

    def tst_try_iter(self):
        return self._xxx_try_iter('tst')


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
            'SpeakerDiarization', 'P1MH', P1MH)

        self.register_protocol(
            'SpeakerDiarization', 'P2', P2)

        self.register_protocol(
            'SpeakerDiarization', 'P2MH', P2MH)

        self.register_protocol(
            'SpeakerSpotting', 'MixHeadset', MixHeadsetSpeakerSpotting)
