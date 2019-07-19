# ODESSA/AMI plugin for pyannote.database

## Installation

```bash
$ pip install pyannote.db.odessa.ami
```

You should then download the dataset files. This repo provides a download script for the required files in the `AMI/db_download/` folder. You can download that file independently from the repository, and run it:

```bash
$ bash ./download.sh /where/you/want/to/download/the/data/
```

You can also download them "by hand" on the [official website](http://groups.inf.ed.ac.uk/ami/download/) by checking all the AMI meetings and only the Headset mix stream.


⚠ The bash download script also "fixes" some of the files from the dataset that are unreadable with scipy because of wrongly formatted wav chunks. The audio files are thus not *exactly* the same as the original one.


Then, tell `pyannote.database` where to look for AMI audio files.

```bash
$ cat ~/.pyannote/database.yml
Databases:
   AMI: /path/to/amicorpus/*/audio/{uri}.wav
```

## Speaker diarization protocol

Protocol is initialized as follows:

```python
>>> from pyannote.database import get_protocol, FileFinder
>>> preprocessors = {'audio': FileFinder()}
>>> protocol = get_protocol('AMI.SpeakerDiarization.MixHeadset',
...                         preprocessors=preprocessors)
```

### Training

For background training (e.g.
[GMM/UBM](http://www.sciencedirect.com/science/article/pii/S1051200499903615), [i-vector](http://ieeexplore.ieee.org/document/5545402/), or [TristouNet neural embedding](https://arxiv.org/abs/1609.04301)), files of the training set can be
iterated using `protocol.train()`:

```python
>>> for current_file in protocol.train():
...
...     # path to the audio file
...     audio = current_file['audio']
...
...     # "who speaks when" reference
...     reference = current_file['annotation']
...
...     # when current_file has an 'annotated' key, it indicates that
...     # annotations are only provided for some regions of the file.
...     annotated = current_file['annotated']
...     # See http://pyannote.github.io/pyannote-core/structure.html#timeline
...
...     # train...
```

`reference` is a pyannote.core.Annotation. In particular, it can be iterated
as follows:

```python
>>> for segment, _, speaker in reference.itertracks(yield_label=True):
...
...     print('Speaker {speaker} speaks between {start}s and {end}s'.format(
...         speaker=speaker, start=segment.start, end=segment.end))
```
See http://pyannote.github.io/pyannote-core/structure.html#annotation for
details on other ways to use this data structure.

### Development

`protocol.development()` is the same as `protocol.train()` except it iterates
over the development set (e.g. for hyper-parameter tuning).

### Test / Evaluation

```python
>>> # initialize evaluation metric
>>> from pyannote.metrics.diarization import DiarizationErrorRate
>>> metric = DiarizationErrorRate()
>>>
>>> # iterate over each file of the test set
>>> for test_file in protocol.test():
...
...     # process test file
...     audio = test_file['audio']
...     hypothesis = process_file(audio)
...
...     # evaluate hypothesis
...     reference = test_file['annotation']
...     uem = test_file['annotated']
...     metric(reference, hypothesis, uem=uem)
>>>
>>> # report results
>>> metric.report(display=True)
```

## Speaker spotting procotol

In order to use the AMI dataset for the evaluation of speaker spotting systems,
the original speaker diarization training/development/test split has been
redefined.

Moreover, original files have also been split into shorter sessions in order to
increase the number of trials.

More details can be found in [this paper](https://www.isca-speech.org/archive/Odyssey_2018/pdfs/60.pdf): 

```bibtex
@inproceedings{Patino2018,
  Title = {{Low-Latency Speaker Spotting with Online Diarization and Detection}},
  Author = {Jose Patino and Ruiqing Yin and H\'{e}ctor Delgado and Herv\'{e} Bredin and Alain Komaty and Guillaume Wisniewski and Claude Barras and Nicholas Evans and S\'{e}bastien Marcel},
  Booktitle = {{Odyssey 2018, The Speaker and Language Recognition Workshop}},
  Pages = {140--146},
  Year = {2018},
  Month = {June},
  Address = {Les Sables d'Olonnes, France},
  url = {http://dx.doi.org/10.21437/Odyssey.2018-20},
}
```

Protocol is initialized as follows:

```python
>>> from pyannote.database import get_protocol, FileFinder
>>> preprocessors = {'audio': FileFinder()}
>>> protocol = get_protocol('AMI.SpeakerSpotting.MixHeadset',
...                         preprocessors=preprocessors)
```


### Training

`protocol.train()` can be used like in the speaker diarization protocol above.

### Enrolment

```python
>>> # dictionary meant to store all target models
>>> models = {}
>>>
>>> # iterate over all enrolments
>>> for current_enrolment in protocol.test_enrolment():
...
...     # target identifier
...     target = current_enrolment['model_id']
...     # the same speaker may be enrolled several times using different target
...     # identifiers. in other words, two different identifiers does not
...     # necessarily not mean two different speakers.
...
...     # path to audio file to use for enrolment
...     audio = current_enrolment['audio']
...
...     # pyannote.core.Timeline containing target speech turns
...     # See http://pyannote.github.io/pyannote-core/structure.html#timeline
...     enrol_with = current_enrolment['enrol_with']
...
...     # this is where enrolment actually happens and model is stored
...     models[target] = enrol(audio, enrol_with)
```

The following pseudo-code shows what the `enrol` function could look like:

```python
>>> def enrol(audio, enrol_with):
...     """Adapt UBM and return GMM"""
...
...     # start by gathering MFCCs
...     mfcc = []
...     for segment in enrol_with:
...         mfcc.append(extract_mfcc(audio, segment.start, segment.end))
...
...     # adapt existing UBM
...     gmm = ubm.adapt(mfcc)
...
...     # return the resulting GMM
...     return gmm
```

### Trial

```python
>>> for current_trial in protocol.test_trial():
...
...     # load target model (precomputed during the enrolment phase)
...     target = current_trial['model_id']
...     model = models[target]
...
...     # path to audio file to use for the trial
...     audio = current_enrolment['audio']

...     # pyannote.core.Segment containing time range where to seek the target
...     # See http://pyannote.github.io/pyannote-core/structure.html#segment
...     try_with = current_enrolment['try_with']
...
...     # this is where speaker spotting actually happens
...     decision = spot(model, audio, try_with)
```

The following pseudo-code shows what the `spot` function could look like:

```python
>>> def spot(model, audio, try_with):
...
...     # perform speaker diarization
...     speakers = speaker_diarization(audio, try_with)
...
...     score = 0.
...     # loop on all speakers
...     for speaker in speakers:
...
...         # gather MFCCs for current speaker
...         mfcc = extract_mfcc(audio, speaker)
...          
...         # compare MFCCs to target model and keep the best score
...         score = max(score, compare(model, mfcc))
...
...     # compare best score to a threshold
...     return score > threshold
```

### Development

`protocol.development()`, `protocol.development_enrolment()`, and
`protocol.development_test()` are also available.

For instance, one could use `protocol.development()` to tune a speaker
diarization module, and `protocol.development_{enrolment|trial}()` to tune
decision thresholds.
