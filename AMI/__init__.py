#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2016-2018 CNRS

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


from pyannote.core import Segment, Timeline, Annotation, SlidingWindow
from pyannote.database import Database
from pyannote.database.protocol import SpeakerDiarizationProtocol
from pyannote.database.protocol import SpeakerSpottingProtocol
from pandas import read_table
from pathlib import Path


class SpeakerDiarization(SpeakerDiarizationProtocol):

    def _load_data(self, subset):

        data_dir = Path(__file__).parent / 'data' / 'speaker_diarization'

        annotated = data_dir / f'{subset}.uem'
        names = ['uri', 'NA0', 'start', 'end']
        annotated = read_table(annotated, delim_whitespace=True, names=names)

        annotation = data_dir / f'{subset}.mdtm'
        names = ['uri', 'NA0', 'start', 'duration',
                 'NA1', 'NA2', 'gender', 'speaker']
        annotation = read_table(annotation, delim_whitespace=True, names=names)

        return {'annotated': annotated,
                'annotation': annotation}

    def _xxx_iter(self, subset):

        data = self._load_data(subset)

        AnnotatedGroups = data['annotated'].groupby(by='uri')
        AnnotationGroups = data['annotation'].groupby(by='uri')

        for raw_uri, annotated in AnnotatedGroups:

            uri = f'{raw_uri}.Mix-Headset'

            segments = []
            for segment in annotated.itertuples():
                segments.append(Segment(start=segment.start, end=segment.end))

            annotation = Annotation(uri=uri)
            for t, turn in enumerate(AnnotationGroups.get_group(raw_uri).itertuples()):
                segment = Segment(start=turn.start,
                                  end=turn.start + turn.duration)
                annotation[segment, t] = turn.speaker

            current_file = {
                'database': 'AMI',
                'uri': uri,
                'annotated': Timeline(uri=uri, segments=segments),
                'annotation': annotation}

            yield current_file

    def trn_iter(self):
        return self._xxx_iter('trn')

    def dev_iter(self):
        return self._xxx_iter('dev')

    def tst_iter(self):
        return self._xxx_iter('tst')


class SpeakerSpotting(SpeakerDiarization, SpeakerSpottingProtocol):

    def _sessionify(self, current_files):

        for current_file in current_files:

            annotated = current_file['annotated']
            annotation = current_file['annotation']

            for segment in annotated:
                sessions = SlidingWindow(start=segment.start,
                                         duration=60., step=60.,
                                         end=segment.end - 60.)

                for session in sessions:

                    session_file = dict(current_file)
                    session_file['annotated'] = annotated.crop(session)
                    session_file['annotation'] = annotation.crop(session)

                    yield session_file

    def trn_iter(self):
        return self._sessionify(super().trn_iter())

    def dev_iter(self):
        return self._sessionify(super().dev_iter())

    def tst_iter(self):
        return self._sessionify(super().tst_iter())

    def _xxx_enrol_iter(self, subset):

        # load enrolments
        data_dir = Path(__file__).parent / 'data' / 'speaker_spotting'
        enrolments = data_dir / f'{subset}.enrol.txt'
        names = ['uri', 'NA0', 'start', 'duration',
                 'NA1', 'NA2', 'NA3', 'model_id']
        enrolments = read_table(enrolments, delim_whitespace=True, names=names)

        for model_id, turns in enrolments.groupby(by='model_id'):

            # gather enrolment data
            segments = []
            for t, turn in enumerate(turns.itertuples()):
                if t == 0:
                    raw_uri = turn.uri
                    uri = f'{raw_uri}.Mix-Headset'
                segment = Segment(start=turn.start,
                                  end=turn.start + turn.duration)
                if segment:
                    segments.append(segment)
            enrol_with = Timeline(segments=segments, uri=uri)

            current_enrolment = {
                'database': 'AMI',
                'uri': uri,
                'model_id': model_id,
                'enrol_with': enrol_with,
            }

            yield current_enrolment

    def dev_enrol_iter(self):
        return self._xxx_enrol_iter('dev')

    def tst_enrol_iter(self):
        return self._xxx_enrol_iter('tst')

    def _xxx_try_iter(self, subset):

        # load "who speaks when" reference and group by (uri, speaker)
        data = self._load_data(subset)
        AnnotationGroups = data['annotation'].groupby(by=['uri', 'speaker'])

        # load trials
        data_dir = Path(__file__).parent / 'data' / 'speaker_spotting'
        trials = data_dir / f'{subset}.trial.txt'
        names = ['model_id', 'uri', 'start', 'end', 'target', 'first', 'total']
        trials = read_table(trials, delim_whitespace=True, names=names)

        for trial in trials.itertuples():

            model_id = trial.model_id

            # FIE038_m1 ==> FIE038
            # FIE038_m42 ==> FIE038
            # Bernard_Pivot_m1 ==> Bernard_Pivot
            speaker = '_'.join(model_id.split('_')[:-1])

            # append Mix-Headset to uri
            raw_uri = trial.uri
            uri = f'{raw_uri}.Mix-Headset'

            # trial session
            session = Segment(start=trial.start, end=trial.end)
            try_with = Timeline(uri=uri, segments=[session])

            # get all turns from target speaker within session
            segments = []
            if trial.target == 'target':
                turns = AnnotationGroups.get_group((raw_uri, speaker))
                for turn in turns.itertuples():
                    segment = Segment(start=turn.start,
                                      end=turn.start + turn.duration)
                    segments.append(segment)
            reference = Timeline(uri=uri, segments=segments).crop(session)

            # pack & yield trial
            current_trial = {
                'database': 'AMI',
                'uri': uri,
                'try_with': try_with,
                'model_id': model_id,
                'reference': reference,
            }

            yield current_trial

    def dev_try_iter(self):
        return self._xxx_try_iter('dev')

    def tst_try_iter(self):
        return self._xxx_try_iter('tst')


class AMI(Database):
    """AMI corpus"""

    def __init__(self, preprocessors={}, **kwargs):
        super(AMI, self).__init__(preprocessors=preprocessors, **kwargs)

        self.register_protocol(
            'SpeakerDiarization', 'MixHeadset', SpeakerDiarization)

        self.register_protocol(
            'SpeakerSpotting', 'MixHeadset', SpeakerSpotting)
