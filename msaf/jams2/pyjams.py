"""JAMS Python API

This library provides an interface for reading JAMS into Python, or creating
them programatically.


1. Creating a JAM from scratch
------------------------------
First, create a top-level Jams object:

  >>> from jams import Jams
  >>> jam = Jams()

Then, you can create an annotation for a given attribute. Here, we'll create a
beat annotation:

  >>> annot = jam.beats.create_annotation()


First, we'll update the annotation's metadata by directly setting its fields:

  >>> annot.annotation_metadata.origin = "Earth"
  >>> annot.annotation_metadata.annotator.email = "grandma@aol.com"


That's not all that interesting though. Now we can populate the annotation:

  >>> beat = annot.create_datapoint()
  >>> beat.time.value = 0.33
  >>> beat.time.confidence = 1.0
  >>> beat.label.value = "1"
  >>> beat.label.confidence = 0.75


And now a second time, cause this is our house (and we can do what we want):

  >>> beat2 = annot.create_datapoint()
  >>> beat2.label.value = "second beat"


Once you've added all your data, you can serialize the annotation to a file
with the built-in `json` library:

  >>> import json
  >>> with open("temp.jams", 'w') as f:
  >>>>>> json.dump(jam, f, indent=2)


2. Reading a Jams file
----------------------
Assuming you already have a JAMS file on-disk, say at 'temp.jams', you can
easily read it back into memory:

  >>> jam2 = J.load('temp.jams')


And that's it!

  >>> print jam2.to_string()

"""

import json
import six

# from time import asctime
from . import __VERSION__


class JSONType(object):
    """Dict-like object for JSON Serialization.

    This object behaves like a dictionary to allow init-level attribute names,
    seamless JSON-serialization, and double-star style unpacking (**obj).
    """
    def __init__(self, **kwargs):
        object.__init__(self)
        for name, value in six.iteritems(kwargs):
            self.__dict__[name] = value

    def __repr__(self):
        """Render the object as an unambiguous string."""
        return '<%s>' % self.__class__.__name__

    def _jsonSupport(*args):
        """TODO(ejhumphrey@nyu.edu): writeme."""
        def default(self, xObject):
            return xObject.__dict__

        json.JSONEncoder.default = default
        json._default_decoder = json.JSONDecoder()

    _jsonSupport()

    def keys(self):
        """TODO(ejhumphrey@nyu.edu): writeme."""
        return self.__dict__.keys()

    def __getitem__(self, key):
        """TODO(ejhumphrey@nyu.edu): writeme."""
        return self.__dict__[key]

    def __len__(self):
        return len(self.keys())

    def update(self, **kwargs):
        for name, value in six.iteritems(kwargs):
            self.__dict__[name] = value

    @property
    def type(self):
        return self.__class__.__name__


class Observation(JSONType):
    """Observation

    Smallest observable concept (value) with a confidence interval. Used for
    almost anything, from observed times to semantic tags.
    """
    def __init__(self, value=None, confidence=0.0):
        """Create an Observation.

        Parameters
        ----------
        value: obj, default=None
            The conceptual value for this observation.
        confidence: float, default=0.0
            Degree of confidence for the value, in the range [0, 1].
        """
        self.value = value
        self.confidence = confidence

    def __repr__(self):
        """Render the object as an unambiguous string."""
        return '<%s: value=%s, confidence=%s>' % \
            (self.type, self.value, self.confidence)


class Label(Observation):
    """Label

    A label is an observation with a specific context.
    E.g.: in music segmentation, a label might have a scope referring to the
        level of the segmentation (functional, large scale, etc.).
    """
    def __init__(self, value=None, confidence=0.0, context=None):
        """Create an Observation.

        Parameters
        ----------
        value: obj, default=None
            The conceptual value for this observation.
        confidence: float, default=0.0
            Degree of confidence for the value, in the range [0, 1].
        context: obj, default=None
            The context of this label, e.g. where it comes from, where it
                belongs.
        """
        super(Label, self).__init__(value, confidence)
        self.context = context

    def __repr__(self):
        """Render the object as an unambiguous string."""
        return '<%s: value=%s, confidence=%s, context=%s>' % \
            (self.type, self.value, self.confidence, self.context)


class Event(Observation):
    """Event (Sparse)

    Instantaneous time event, consisting of two Observations (time and label).
    Used for such ideas like beats or onsets.
    """
    def __init__(self, time=None, label=None):
        """Create an Event.

        Note that, if an argument is None, an empty Observation is created in
        its place. Additionally, a dictionary matching the expected structure
        of the arguments will be parsed successfully (i.e. instantiating from
        JSON).

        Parameters
        ----------
        time: Observation (or dict), default=None
            A time Observation for this event.
        label: Observation (or dict), default=None
            A semantic concept for this event, as an Observation.
        """
        if time is None:
            time = Observation()
        if label is None:
            label = Label()
        self.time = Observation(**time)
        self.label = Label(**label)

    def __repr__(self):
        """Render the object as an unambiguous string."""
        return '<%s:\n\ttime=%s,\n\tlabel=%s>' % \
            (self.type, self.time, self.label)


class Range(Observation):
    """Range

    An observed time interval, composed of three Observations (start, end, and
    label). Used for such concepts as chords.
    """
    def __init__(self, start=None, end=None, label=None):
        """Create a Range.

        Note that, if an argument is None, an empty Observation is created in
        its place. Additionally, a dictionary matching the expected structure
        of the arguments will be parsed successfully (i.e. instantiating from
        JSON).

        Parameters
        ----------
        start: Observation (or dict)
            The observed start time of the range.
        end: Observation (or dict)
            The observed end time of the range.
        label: Observation (or dict)
            Label over this time interval.
        """
        if start is None:
            start = Observation()
        if end is None:
            end = Observation()
        if label is None:
            label = Label()
        self.start = Observation(**start)
        self.end = Observation(**end)
        self.label = Label(**label)

    @property
    def duration(self):
        return self.end.value - self.start.value

    def __repr__(self):
        """Render the object as an unambiguous string."""
        return '<%s:\n\tstart=%s,\n\tend=%s,\n\tlabel=%s>' % \
            (self.type, self.start, self.end, self.label)


class TimeSeries(Observation):
    """Sampled Time Series Observation

    This could be an array, and skip the value abstraction. However,
    some abstraction could help turn data into numpy arrays on the fly.

    However, np.ndarrays are not directly serializable. It might be necessary
    to subclass np.ndarray and change __repr__.
    """
    def __init__(self, value=None, time=None, confidence=None):
        """Create a TimeSeries.

        Note that, if an argument is None, empty lists are created in
        its place. Additionally, a dictionary matching the expected structure
        of the arguments will be parsed successfully (i.e. instantiating from
        JSON).

        Parameters
        ----------
        value: list or serializable array
            Values for this time-series.
        time: list or serializable 1D-array
            Times corresponding to the value series.
        confidence: list or serializable 1D-array
            Confidence values corresponding to the value series.
        """
        if value is None:
            value = list()
        if time is None:
            time = list()
        if confidence is None:
            confidence = list()
        self.value = value
        self.time = time
        self.confidence = confidence


class Annotation(JSONType):
    """Annotation base class.

    Default Type: Observation

    Be aware that Annotations define a '_DefaultType' class variable,
    specifying the kind of objects contained in its 'data' attribute. Therefore
    any subclass will need to set this accordingly.
    """
    _DefaultType = Observation

    def __init__(self, data=None, annotation_metadata=None, sandbox=None):
        """Create an Annotation.

        Note that, if an argument is None, an empty Annotation is created in
        its place. Additionally, a dictionary matching the expected structure
        of the arguments will be parsed successfully (i.e. instantiating from
        JSON).

        Parameters
        ----------
        data: list, or None
            Collection of Observations
        annotation_metadata: AnnotationMetadata (or dict), default=None.
            Metadata corresponding to this Annotation.
        """
        # TODO(ejhumphrey@nyu.edu): We may want to subclass list here to turn
        #   'data' into a special container with convenience methods to more
        #   easily unpack sparse events, among other things.
        if data is None:
            data = list()
        if annotation_metadata is None:
            annotation_metadata = AnnotationMetadata()
        if sandbox is None:
            sandbox = JSONType()
        self.data = self.__parse_data__(data)
        self.annotation_metadata = AnnotationMetadata(**annotation_metadata)
        self.sandbox = JSONType(**sandbox)

    def __parse_data__(self, data):
        """This method unpacks data as a specific type of objects, defined by
        the self._DefaultType, for the purposes of safely creating a list of
        properly initialized objects.

        Parameters
        ----------
        data: list
            Collection of dicts or _DefaultTypes.

        Returns
        -------
        objects: list
            Collection of _DefaultTypes.
        """
        return [self._DefaultType(**obj) for obj in data]

    def create_datapoint(self):
        """Create an empty Data object based on this type of Annotation, adding
        it to the data list and returning a reference.

        Returns
        -------
        obj: self._DefaultType
            An empty object, whose type is determined by the Annotation type.
        """
        self.data.append(self._DefaultType())
        return self.data[-1]


class EventAnnotation(Annotation):
    """Event Annotation

    Default Type: Event

    Be aware that Annotations define a '_DefaultType' class variable,
    specifying the kind of objects contained in its 'data' attribute. Therefore
    any subclass will need to set this accordingly."""
    _DefaultType = Event

    @property
    def labels(self):
        """All labels in the annotation.

        Returns
        -------
        labels: list of tuples
            Ordered collection of (value, confidence) pairs over all labels.
        """
        return [(obj.label.value, obj.label.confidence) for obj in self.data]

    @property
    def times(self):
        """All times in the annotation.

        Returns
        -------
        times: list of tuples
            Ordered collection of (value, confidence) pairs over all times.
        """
        return [(obj.time.value, obj.time.confidence) for obj in self.data]


class TimeSeriesAnnotation(Annotation):
    """TimeSeries Annotation

    Default Type: TimeSeries

    Be aware that Annotations define a '_DefaultType' class variable,
    specifying the kind of objects contained in its 'data' attribute. Therefore
    any subclass will need to set this accordingly."""
    _DefaultType = TimeSeries


class RangeAnnotation(Annotation):
    """Range Annotation

    Default Type: Range

    Be aware that Annotations define a '_DefaultType' class variable,
    specifying the kind of objects contained in its 'data' attribute. Therefore
    any subclass will need to set this accordingly."""
    _DefaultType = Range

    @property
    def labels(self):
        """All labels in the annotation.

        Returns
        -------
        labels: list of tuples
            Ordered collection of (value, confidence, context) triplets.
        """
        return [(obj.label.value, obj.label.confidence, obj.label.context)
                for obj in self.data]

    @property
    def starts(self):
        """All start times in the annotation.

        Returns
        -------
        starts: list of tuples
            Ordered collection of (value, confidence) pairs.
        """
        return [(obj.start.value, obj.start.confidence) for obj in self.data]

    @property
    def ends(self):
        """All end times in the annotation.

        Returns
        -------
        ends: list of tuples
            Ordered collection of (value, confidence) pairs.
        """
        return [(obj.end.value, obj.end.confidence) for obj in self.data]

    @property
    def boundaries(self):
        """All start and end times in the annotation.

        Returns
        -------
        times: list of tuples
            Ordered collection of (start.value, end.value) pairs/
        """
        return [(obj.start.value, obj.end.value) for obj in self.data]


class AnnotationSet(list):
    """AnnotationSet

    List subclass for managing collections of annotations, providing the
    functionality to create empty annotations.
    """
    def __init__(self, annotations=None, DefaultType=Annotation):
        """Create an AnnotationSet.

        Note that using the default arguments will create an empty
        AnnotationSet.

        Parameters
        ----------
        annotations: list, or None
            Collection of Annotations, or appropriately formated dicts.
        DefaultType: Annotation, or subclass
            Class to use as a default type in this object.
        """
        self._DefaultType = DefaultType
        self.extend([self._DefaultType(**obj) for obj in annotations])

    def create_annotation(self):
        """Create an empty Annotation based on the DefaultType passed on init,
        adding it to the annotation list and returning a reference.

        Returns
        -------
        obj: Annotation, or subclass
            An empty annotation, whose type is determined by self._DefaultType.
        """
        self.append(self._DefaultType())
        return self[-1]


class Annotator(JSONType):
    """Annotator

    Container object for annotator metadata.
    """
    def __init__(self, name='', email=''):
        """Create an Annotator.

        Parameters
        ----------
        name: str, default=''
            Common name of the annotator.
        email: str, default=''
            An email address corresponding to the annotator.
        """
        self.name = name
        self.email = email


class AnnotationMetadata(JSONType):
    """AnnotationMetadata

    Data structure for metadata corresponding to a specific annotation.

    Note: We *desperately* need to rename some of these properties; certain
    names are far too verbose.
    """
    def __init__(self, attribute='', corpus='', version=0, annotator=None,
                 annotation_tools='', annotation_rules='',
                 validation_and_reliability='', origin=''):
        """Create an AnnotationMetadata object.

        Parameters
        ----------
        attribute: str, default=''
            Attribute type, e.g. beats, chords, etc. *Needed?
        corpus: str, default=''
            Collection assignment.
        version: scalar, default=0
            Version number.
        annotator: Annotator, default=None
            Annotator object, empty if none is specified.
        annotation_tools: str, default=''
            Description of the tools used to create the annotation.
        annotation_rules: str, default=''
            Description of the rules provided to the annotator.
        validation_and_reliability: str, default=''
            TODO(justin.salamon@nyu.edu): What is this?
        origin: str, default=''
            From whence it came.

        -- Also add? --
        if timestamp is None:
            timestamp = asctime()
        """
        if annotator is None:
            annotator = Annotator()
        self.attribute = attribute
        self.corpus = corpus
        self.version = version
        self.annotator = Annotator(**annotator)
        self.annotation_tools = annotation_tools
        self.annotation_rules = annotation_rules
        self.validation_and_reliability = validation_and_reliability
        self.origin = origin


class Metadata(JSONType):
    """Metadata

    Data structure for file-level metadata.
    """
    def __init__(self, title='', artist='', md5='', duration='',
                 echonest_id='', mbid='', version=None):
        """Create a file-level Metadata object.

        Parameters
        ----------
        title: str
            Name of the recording.
        artist: str
            Name of the artist / musician.
        md5: str
            MD5 hash of the corresponding file.
        duration: str
            Time duration of the file, as HH:MM:SS.
        echonest_id: str
            Echonest ID for this track.
        mbid: str
            MusicBrainz ID for this track.
        version: str, or default=None
            Version of the JAMS Schema.
        """
        if version is None:
            version = __VERSION__
        self.title = title
        self.artist = artist
        self.md5 = md5
        self.duration = duration
        self.echonest_id = echonest_id
        self.mbid = mbid
        self.version = version

    def __eq__(self, y):
        """Test for equality between two metadata-like dictionaries."""
        return dict(**self) == dict(**y)


class Jams(JSONType):
    """Top-level Jams Object

    Note: I don't want to leave this named "Jams", it looks really freakin
    weird. Not sure what else the top-level object should be called though.
    """
    def __init__(self, beats=None, chords=None, melody=None, notes=None,
                 pitch=None, sections=None, sources=None, tags=None,
                 metadata=None, sandbox=None):
        """Create a Jams object.

        Parameters
        ----------
        beats: list of EventAnnotations
            Used for beat-tracking.
        chords: list of RangeAnnotations
            Used for chord recognition.
        melody: list of TimeSeriesAnnotations
            Used for continuous-f0 melody.
        sections: list of RangeAnnotations
            Used for structural analysis, as sections / segmentation.
        tags: list of Annotations
            Used for music tagging and semantic descriptors.
        metadata: Metadata
            Metadata corresponding to the audio file.
        """
        if beats is None:
            beats = list()
        if chords is None:
            chords = list()
        if melody is None:
            melody = list()
        if notes is None:
            notes = list()
        if pitch is None:
            pitch = list()
        if sections is None:
            sections = list()
        if sources is None:
            sources = list()
        if tags is None:
            tags = list()
        if metadata is None:
            metadata = Metadata()
        if sandbox is None:
            sandbox = JSONType()

        self.beats = AnnotationSet(beats, EventAnnotation)
        self.chords = AnnotationSet(chords, RangeAnnotation)
        self.melody = AnnotationSet(melody, TimeSeriesAnnotation)
        self.notes = AnnotationSet(notes, RangeAnnotation)
        self.pitch = AnnotationSet(pitch, TimeSeriesAnnotation)
        self.sections = AnnotationSet(sections, RangeAnnotation)
        self.sources = AnnotationSet(sources, RangeAnnotation)
        self.tags = AnnotationSet(tags, Annotation)
        self.metadata = Metadata(**metadata)
        self.sandbox = JSONType(**sandbox)

    def add(self, jam, on_conflict='fail'):
        """Add the contents of another jam to this object.

        Note that this method fails if the top-level metadata is not identical,
        raising a ValueError; either resolve this manually (because conflicts
        should almost never happen), force an 'overwrite', or tell the method
        to 'ignore' the metadata of the object being added.

        Parameters
        ----------
        jam: Jams object
            Object to add to this jam
        on_conflict: str, default='fail'
            Strategy for resolving metadata conflicts; one of
                ['fail', 'overwrite', or 'ignore'].
        """
        if on_conflict == 'overwrite':
            self.metadata = jam.metadata
        elif on_conflict == 'fail' and not self.metadata == jam.metadata:
            raise ValueError("Metadata conflict! "
                             "Resolve manually or force-overwrite it.")
        elif on_conflict == 'ignore':
            pass
        else:
            raise ValueError("on_conflict received '%s'. Must be one of "
                             "['fail', 'overwrite', 'ignore']." % on_conflict)

        self.beats.extend(jam.beats)
        self.chords.extend(jam.chords)
        self.melody.extend(jam.melody)
        self.notes.extend(jam.notes)
        self.pitch.extend(jam.pitch)
        self.sections.extend(jam.sections)
        self.sources.extend(jam.sources)
        self.tags.extend(jam.tags)
        self.sandbox.update(**jam.sandbox)

    @classmethod
    def from_string(cls, jam_str):
        """Alternate constructor. Create an object from a JAMS string.

        Parameters
        ----------
        jam_str: str
            String representation of a JAMS object.
        """
        return cls(**json.loads(jam_str))

    def to_string(self, indent=2):
        """Render the JAMS object as a JSON-string."""
        # TODO(ejhumphrey): Only dump populated fields.
        return json.dumps(self, indent=indent)
