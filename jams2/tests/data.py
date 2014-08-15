"""Utilities to create random Jams objects.
"""

from .. import pyjams

import random
import string

ASCII_SET = string.ascii_letters + ' '*10 + string.digits


def random_string(low=4, high=20):
    return "".join(
        [random.choice(ASCII_SET) for n in xrange(random.randint(low, high))])


def random_Observation():
    """Populate an Observation with random values.
    """
    x = pyjams.Observation()

    x.value = random_string(4, 20)
    x.confidence = random.uniform(0, 1.0)
    return x


def random_Label():
    """Populate a Label with random values.
    """
    x = pyjams.Label()
    x.value = random_string(4, 20)
    x.context = random_string(4, 20)
    x.confidence = random.uniform(0, 1.0)
    return x


def random_Event(max_time=300.0):
    """Populate an Event with random values.
    """
    x = pyjams.Event()
    x.time.value = random.uniform(0, max_time)
    x.time.confidence = random.uniform(0, 1.0)
    x.label.value = random_string(4, 20)
    x.label.context = random_string(4, 20)
    x.label.confidence = random.uniform(0, 1.0)
    return x


def random_Range(max_time=300.0):
    """Populate a Range with random values.
    """
    x = pyjams.Range()
    times = [random.uniform(0, max_time) for n in range(2)]
    times.sort()

    x.start.value = times[0]
    x.start.confidence = random.uniform(0, 1.0)
    x.end.value = times[1]
    x.end.confidence = random.uniform(0, 1.0)
    x.label.value = random_string(4, 20)
    x.label.context = random_string(4, 20)
    x.label.confidence = random.uniform(0, 1.0)
    return x


def random_TimeSeries(min_length=200, max_length=1000):
    """Populate a TimeSeries with random values.
    """
    x = pyjams.TimeSeries()
    N = random.randrange(min_length, max_length)
    x.value = [random.uniform(0, 1000) for n in range(N)]
    x.time = [random.uniform(0, 1.0) for n in range(N)]
    x.time.sort()
    x.confidence = [random.uniform(0, 1.0) for n in range(N)]
    return x


def random_AnnotationMetadata():
    metadata = pyjams.AnnotationMetadata()
    metadata.attribute = random_string()
    metadata.corpus = random_string()
    metadata.version = random_string()
    metadata.annotator = pyjams.Annotator(random_string(), random_string())
    metadata.annotation_tools = random_string()
    metadata.annotation_rules = random_string()
    metadata.validation_and_reliability = random_string()
    metadata.origin = random_string()
    return metadata


def random_Annotation(num_items):
    return pyjams.Annotation(
        data=[random_Observation() for n in range(num_items)],
        annotation_metadata=random_AnnotationMetadata())


def random_EventAnnotation(num_items, max_time=300):
    return pyjams.EventAnnotation(
        data=[random_Event(max_time=max_time) for n in range(num_items)],
        annotation_metadata=random_AnnotationMetadata())


def random_RangeAnnotation(num_items, max_time=300):
    return pyjams.RangeAnnotation(
        data=[random_Range(max_time=max_time) for n in range(num_items)],
        annotation_metadata=random_AnnotationMetadata())


def random_TimeSeriesAnnotation(num_items, min_length=200, max_length=1000):
    return pyjams.TimeSeriesAnnotation(
        data=[random_TimeSeries(min_length=min_length, max_length=max_length)
              for n in range(num_items)],
        annotation_metadata=random_AnnotationMetadata())


def random_Metadata():
    return pyjams.Metadata(*[random_string() for n in range(7)])


def random_Jams(beats=(), chords=(), melody=(), notes=(),
                pitch=(), sections=(), sources=(), tags=()):
    return pyjams.Jams(
        beats=[random_EventAnnotation(*args) for args in beats],
        chords=[random_RangeAnnotation(*args) for args in chords],
        melody=[random_TimeSeriesAnnotation(*args) for args in melody],
        notes=[random_RangeAnnotation(*args) for args in notes],
        pitch=[random_TimeSeriesAnnotation(*args) for args in pitch],
        sections=[random_RangeAnnotation(*args) for args in sections],
        sources=[random_RangeAnnotation(*args) for args in sources],
        tags=[random_Annotation(*args) for args in tags],
        metadata=random_Metadata())
