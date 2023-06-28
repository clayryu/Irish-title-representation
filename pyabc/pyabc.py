from __future__ import division
import re, sys


str_type = str if sys.version > '3' else basestring


# Just some tunes to test against
tunes = [
"""
X: 1
T: The Road To Lisdoonvarna
R: slide
M: 12/8
L: 1/8
K: Edor
E2B B2A B2c d2A|F2A ABA D2E FED|E2B B2A B2c d3|cdc B2A B2E E3:|
|:e2f gfe d2B Bcd|c2A ABc d2B B3|e2f gfe d2B Bcd|cdc B2A B2E E3:||
""",
"""
X: 6
T: The Kid On The Mountain
R: slip jig
M: 9/8
L: 1/8
K: Emin
~E3 FEF G2 F| ~E3 BcA BGD| ~E3 FEF G2 A| BAG FAG FED:|
BGB AFD G2 D| GAB dge dBA| BGB AFA G2 A| BAG FAG FED:|
~g3 eBe e2 f|~g3 efg afd| ~g3 eBe g2 a|bag fag fed:|
eB/B/B e2f ~g3|eB/B/B efg afd| eB/B/B e2f g2a|bag fag fed:|
edB dBA G2D|GAB dge dBA|edB dBA G2A|BAG FAG FED:|
"""
]

# Information field table copied from
# http://abcnotation.com/wiki/abc:standard:v2.1#abc_files_tunes_and_fragments
# Columns are:
# X:Field, file header, tune header, tune body, inline, type
information_field_table = """
A:area              yes     yes     no      no      string
B:book              yes     yes     no      no      string
C:composer          yes     yes     no      no      string
D:discography       yes     yes     no      no      string
F:file url          yes     yes     no      no      string
G:group             yes     yes     no      no      string
H:history           yes     yes     no      no      string
I:instruction       yes     yes     yes     yes     instruction
K:key               no      yes     yes     yes     instruction
L:unit note length  yes     yes     yes     yes     instruction
M:meter             yes     yes     yes     yes     instruction
m:macro             yes     yes     yes     yes     instruction
N:notes             yes     yes     yes     yes     string
O:origin            yes     yes     no      no      string
P:parts             no      yes     yes     yes     instruction
Q:tempo             no      yes     yes     yes     instruction
R:rhythm            yes     yes     yes     yes     string
r:remark            yes     yes     yes     yes     -
S:source            yes     yes     no      no      string
s:symbol line       no      no      yes     no      instruction
T:tune title        no      yes     yes     no      string
U:user defined      yes     yes     yes     yes     instruction
V:voice             no      yes     yes     yes     instruction
W:words             no      yes     yes     no      string
w:words             no      no      yes     no      string
X:reference number  no      yes     no      no      instruction
Z:transcription     yes     yes     no      no      string
"""



class InfoKey(object):
    def __init__(self, key, name, file_header, tune_header, tune_body, inline, type):
        self.key = key  # single-letter field identifier
        self.name = name.strip()  # information field name
        self.file_header = file_header=='yes'  # may be used in file header
        self.tune_header = tune_header=='yes'  # may be used in tune header
        self.tune_body = tune_body=='yes'  # may be used in tune body
        self.inline = inline=='yes'  # nay be used inline in tunes
        self.type = type.strip()  # data type:  string, instruction, or -

# parse info field table
info_keys = {}
for line in information_field_table.split('\n'):
    if line.strip() == '':
        continue
    key = line[0]
    fields = re.match(r'(.*)\s+(yes|no)\s+(yes|no)\s+(yes|no)\s+(yes|no)\s+(.*)', line[2:]).groups()
    info_keys[key] = InfoKey(key, *fields)

file_header_fields = {k:v for k,v in info_keys.items() if v.file_header}
tune_header_fields = {k:v for k,v in info_keys.items() if v.tune_header}
tune_body_fields = {k:v for k,v in info_keys.items() if v.tune_body}
inline_fields = {k:v for k,v in info_keys.items() if v.inline}



# map natural note letters to chromatic values
pitch_values = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11, }
accidental_values = {'': 0, '#': 1, 'b': -1}
for n,v in list(pitch_values.items()):
    for a in '#b':
        pitch_values[n+a] = v + accidental_values[a]

# map chromatic number back to most common key names
chromatic_notes = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']

# map mode names relative to Ionian (in chromatic steps)
mode_values = {'major': 0, 'minor': 3, 'ionian': 0, 'aeolian': 3,
               'mixolydian': -7, 'dorian': -2, 'phrygian': -4, 'lydian': -5,
               'locrian': 1}

# mode name normalization
mode_abbrev = {m[:3]: m for m in mode_values}

# sharps/flats in ionian keys
key_sig = {'C#': 7, 'F#': 6, 'B': 5, 'E': 4, 'A': 3, 'D': 2, 'G': 1, 'C': 0,
           'F': -1, 'Bb': -2, 'Eb': -3, 'Ab': -4, 'Db': -5, 'Gb': -6, 'Cb': -7}
sharp_order = "FCGDAEB"
flat_order = "BEADGCF"


class Key(object):
    def __init__(self, name=None, root=None, mode=None):
        if name is not None:
            self.root, self.mode = self.parse_key(name)
            assert root is None and mode is None
        else:
            self.root = Pitch(root)
            self.mode = mode

    def parse_key(self, key):
        # highland pipe keys
        if key in ['HP', 'Hp']:
            return {'F': 1, 'C': 1, 'G': 0}

        m = re.match(r'([A-G])(\#|b)?\s*(\w+)?(.*)', key)
        if m is None:
            raise ValueError('Invalid key "%s"' % key)
        base, acc, mode, extra = m.groups()
        if acc is None:
            acc = ''
        if mode is None:
            mode = 'major'
        if mode == 'm':
            mode = 'minor'
        try:
            mode = mode_abbrev[mode[:3].lower()]
        except KeyError:
            raise ValueError("Unrecognized key signature %s" % key)

        return Pitch(base+acc), mode

    @property
    def key_signature(self):
        """
        List of accidentals that should be displayed in the key
        signature for the given key description.
        """
        # determine number of sharps/flats for this key by first converting
        # to ionian, then doing the key lookup
        key = self.relative_ionian
        num_acc = key_sig[key.root.name]

        sig = []
        # sharps or flats?
        if num_acc > 0:
            for i in range(num_acc):
                sig.append(sharp_order[i] + '#')
        else:
            for i in range(-num_acc):
                sig.append(flat_order[i] + 'b')

        return sig

    @property
    def accidentals(self):
        """A dictionary of accidentals in the key signature.
        """
        return {p:a for p,a in self.key_signature}

    @property
    def relative_ionian(self):
        """
        Return the ionian mode relative to the given key and mode.
        """
        key, mode = self.root, self.mode
        rel = mode_values[mode]
        root = Pitch((key.value + rel) % 12)

        # Select flat or sharp to match the current key name
        if '#' in key.name:
            root2 = root.equivalent_sharp
            if len(root2.name) == 2:
                root = root2
        elif 'b' in key.name:
            root2 = root.equivalent_flat
            if len(root2.name) == 2:
                root = root2

        return Key(root=root, mode='ionian')

    def __repr__(self):
        return "<Key %s %s>" % (self.root.name, self.mode)


class Pitch(object):
    def __init__(self, value, octave=None):
        if isinstance(value, Note):
            self._note = value

            if len(value.note) == 1:
                acc = value.key.accidentals.get(value.note[0].upper(), '')
                self._name = value.note.upper() + acc
                self._value = self.pitch_value(self._name)
            else:
                self._name = value.note.capitalize()
                self._value = self.pitch_value(value.note)

            assert octave is None
            self._octave = value.octave
        elif isinstance(value, str_type):
            self._name = value
            self._value = self.pitch_value(value)
            self._octave = octave
        elif isinstance(value, Pitch):
            self._name = value._name
            self._value = value._value
            self._octave = value._octave
        else:
            self._name = None
            if octave is None:
                self._value = value
                self._octave = octave
            else:
                self._value = value % 12
                self._octave = octave + (value // 12)

    def __repr__(self):
        return "<Pitch %s>" % self.name

    @property
    def name(self):
        if self._name is not None:
            return self._name
        return chromatic_notes[self.value%12]

    @property
    def value(self):
        return self._value

    @property
    def octave(self):
        return self._octave

    @property
    def abs_value(self):
        return self.value + self.octave * 12

    @staticmethod
    def pitch_value(pitch, root='C'):
        """Convert a pitch string like "A#" to a chromatic scale value relative
        to root.
        """
        pitch = pitch.strip()
        val = pitch_values[pitch[0].upper()]
        for acc in pitch[1:]:
            val += accidental_values[acc]
        if root == 'C':
            return val
        return (val - Pitch.pitch_value(root)) % 12

    def __eq__(self, a):
        return self.value == a.value

    @property
    def equivalent_sharp(self):
        p = self - 1
        if len(p.name) == 1:
            return Pitch(p.name + '#', octave=self.octave)
        else:
            return Pitch((self-2).name + '##', octave=self.octave)

    @property
    def equivalent_flat(self):
        p = self + 1
        if len(p.name) == 1:
            return Pitch(p.name + 'b', octave=self.octave)
        else:
            return Pitch((self+2).name + 'bb', octave=self.octave)

    def __add__(self, x):
        return Pitch(self.value+x, octave=self.octave)

    def __sub__(self, x):
        return Pitch(self.value-x, octave=self.octave)


class TimeSignature(object):
    def __init__(self, meter, unit_len, tempo=None):
        meter = meter.replace('C|', '2/2').replace('C', '4/4')
        self._meter = [int(x) for x in meter.split('/')]
        self._unit_len = [int(x) for x in unit_len.split('/')]
        self._tempo = tempo

    def __repr__(self):
        return "<TimeSignature %d/%d>" % tuple(self._meter)


# Decoration symbols from
# http://abcnotation.com/wiki/abc:standard:v2.1#decorations
symbols = """
!trill!                "tr" (trill mark)
!trill(!               start of an extended trill
!trill)!               end of an extended trill
!lowermordent!         short /|/|/ squiggle with a vertical line through it
!uppermordent!         short /|/|/ squiggle
!mordent!              same as !lowermordent!
!pralltriller!         same as !uppermordent!
!roll!                 a roll mark (arc) as used in Irish music
!turn!                 a turn mark (also known as gruppetto)
!turnx!                a turn mark with a line through it
!invertedturn!         an inverted turn mark
!invertedturnx!        an inverted turn mark with a line through it
!arpeggio!             vertical squiggle
!>!                    > mark
!accent!               same as !>!
!emphasis!             same as !>!
!fermata!              fermata or hold (arc above dot)
!invertedfermata!      upside down fermata
!tenuto!               horizontal line to indicate holding note for full duration
!0! - !5!              fingerings
!+!                    left-hand pizzicato, or rasp for French horns
!plus!                 same as !+!
!snap!                 snap-pizzicato mark, visually similar to !thumb!
!slide!                slide up to a note, visually similar to a half slur
!wedge!                small filled-in wedge mark
!upbow!                V mark
!downbow!              squared n mark
!open!                 small circle above note indicating open string or harmonic
!thumb!                cello thumb symbol
!breath!               a breath mark (apostrophe-like) after note
!pppp! !ppp! !pp! !p!  dynamics marks
!mp! !mf! !f! !ff!     more dynamics marks
!fff! !ffff! !sfz!     more dynamics marks
!crescendo(!           start of a < crescendo mark
!<(!                   same as !crescendo(!
!crescendo)!           end of a < crescendo mark, placed after the last note
!<)!                   same as !crescendo)!
!diminuendo(!          start of a > diminuendo mark
!>(!                   same as !diminuendo(!
!diminuendo)!          end of a > diminuendo mark, placed after the last note
!>)!                   same as !diminuendo)!
!segno!                2 ornate s-like symbols separated by a diagonal line
!coda!                 a ring with a cross in it
!D.S.!                 the letters D.S. (=Da Segno)
!D.C.!                 the letters D.C. (=either Da Coda or Da Capo)
!dacoda!               the word "Da" followed by a Coda sign
!dacapo!               the words "Da Capo"
!fine!                 the word "fine"
!shortphrase!          vertical line on the upper part of the staff
!mediumphrase!         same, but extending down to the centre line
!longphrase!           same, but extending 3/4 of the way down
"""



class Token(object):
    def __init__(self, line, char, text):
        self._line = line
        self._char = char
        self._text = text
        self.tune_offset = None
        self.measure_offset = None
        self.measure_number = None
        self.meas_offset_from_repeat_start = None

    def __repr__(self):
        return "<%s \"%s\">" % (self.__class__.__name__, self._text)


class Note(Token):
    def __init__(self, key, time, note, accidental, octave, num, denom, **kwds):
        Token.__init__(self, **kwds)
        self.key = key
        self.time_sig = time
        self.note = note
        self.accidental = accidental
        self.octave = octave
        self._length = (num, denom)

        self.midi_pitch = self.pitch.abs_value + 4 * 12

    @property
    def pitch(self):
        """Chromatic note value taking into account key signature and transpositions.
        """
        return Pitch(self)

    @property
    def length(self):
        n,d = self._length
        return (int(n) if n is not None else 1, int(d) if d is not None else 1)

    @property
    def duration(self):
        return self.length[0] / self.length[1]

    def dotify(self, dots, direction):
        """Apply dot(s) to the duration of this note.
        """
        assert direction in ('left', 'right')
        longer = direction == 'left'
        if '<' in dots:
            longer = not longer
        n_dots = len(dots)
        num, den = self.length
        if longer:
            num = num * 2 + 1
            den = den * 2
            self._length = (num, den)
        else:
            den = den * 2
            self._length = (num, den)



class Beam(Token):
    pass

class Space(Token):
    pass

class Slur(Token):
    """   ( or )   """
    pass

class Tie(Token):
    """   -   """
    pass

class Newline(Token):
    pass

class Continuation(Token):
    """  \\ at end of line  """
    pass

class GracenoteBrace(Token):
    """  {  {/  or }  """
    pass

class ChordBracket(Token):
    """  [  or  ]  """
    pass

class ChordSymbol(Token):
    """   "Amaj"   """
    pass

class Annotation(Token):
    """    "<stuff"   """
    pass

class Decoration(Token):
    """  .~HLMOPSTuv  """
    pass

class Tuplet(Token):
    """  (5   """
    def __init__(self, num, **kwds):
        Token.__init__(self, **kwds)
        self.num = num

class BodyField(Token):
    pass

class InlineField(Token):
    pass

class Rest(Token):
    def __init__(self, symbol, num, denom, **kwds):
        # char==X or Z means length is in measures
        Token.__init__(self, **kwds)
        self.symbol = symbol
        self._length = (num, denom)

    @property
    def length(self):
        n,d = self._length
        return (int(n) if n is not None else 1, int(d) if d is not None else 1)

    @property
    def duration(self):
        return self.length[0] / self.length[1]

    def dotify(self, dots, direction):
        """Apply dot(s) to the duration of this note.
        """
        assert direction in ('left', 'right')
        longer = direction == 'left'
        if '<' in dots:
            longer = not longer
        n_dots = len(dots)
        num, den = self.length
        if longer:
            num = num * 2 + 1
            den = den * 2
            self._length = (num, den)
        else:
            den = den * 2
            self._length = (num, den)

class InfoContext(object):
    """Keeps track of current information fields
    """
    def __init__(self, fields):
        self._fields = fields

    def __getattr__(self, field):
        return self._fields.get(field, None)

    def copy(self, fields):
        """Return a copy with some fields updated
        """
        f2 = InfoContext(self._fields)
        f2._fields.update(fields)
        return f2


class Measure(object):
  def __init__(self, number, tokens, beam_text, full_duration):
    self.number = number
    self.meas_offset_from_repeat_start = 0
    self.tokens = tokens
    self.text = beam_text
    self.prev_beam_text = ''
    self.duration = 0
    self.tune_offset = None

    self.duration, self.offset_by_token = self._parse_duration()
    self.full_duration = full_duration
    if self.duration != self.full_duration:
      self.is_incomplete = True
    else:
      self.is_incomplete = False
    self._update_meas_offset()

  def _parse_duration(self):
    measure_duration = 0
    offset_by_token = []
    tuplet_count = 0
    tuplet_duration = 0
    gracenoteflag = False
    chordbracketflag = False
    chordbracketduration = 0

    for idx, token in enumerate(self.tokens):
      offset_by_token.append(measure_duration)
      if isinstance(token, Space):
        continue
      elif isinstance(token, Newline):
        continue
      elif isinstance(token, Slur):
        continue
      elif isinstance(token, Annotation):
        continue
      elif isinstance(token, Decoration):
        continue

        
      #박자와 맞는지 확인,
      elif isinstance(token, Note):
        if tuplet_count > 0:
          tuplet_duration = token.duration
          tuplet_count = tuplet_count - 1
          if tuplet_count == 0:
            measure_duration += tuplet_duration * 2
        elif gracenoteflag == True:
          continue
        elif chordbracketflag == True:
          chordbracketduration = token.duration
        else:
          measure_duration += token.duration
          # duration_list.append(f"{token.duration}, {idx}")
            
            
      elif isinstance(token, Tuplet):
        tuplet_count = int(token.num)
        
      elif isinstance(token, GracenoteBrace):
        if token._text == '{':
          gracenoteflag = True
        elif token._text == '}':
          gracenoteflag = False

      elif isinstance(token, Rest):
        if token.length[0] == None:
          measure_duration+=1
        else:
          measure_duration += int(token.length[0])
            
      elif isinstance(token, ChordBracket):
        if token._text == '[':
          chordbracketflag = True
        elif ']' in token._text:
          chordbracketflag = False
          measure_duration += chordbracketduration
    return measure_duration, offset_by_token

  def _update_meas_offset(self):
    for token, offset in zip(self.tokens, self.offset_by_token):
      token.measure_offset = offset
      token.measure_number = self.number
    
  def __repr__(self):
    return f"Measure Number: {self.number}, Bar type: {self.text}, Duration: {self.duration}, Tokens: {self.tokens}"

class Tunes:
    def __init__(self, abc=None):
        self.num_subtunes = 0
        if abc is not None:
            split_abc = self.split_abc_tunes(abc)
            self.tunes = []
            for sub_abc in split_abc:
              try:
                self.tunes.append(Tune(sub_abc))
              except Exception as e:
                # print(f"Error while reading tune: {e}")
                continue
            # self.tunes = [Tune(sub_abc) for sub_abc in split_abc]
        else:
            raise TypeError("must provide abc or json")

    def split_abc_tunes(self, abc):
      split_abc = abc.split('\nX')
      new_list = []
      for subtune in split_abc:
        if subtune[0] != 'X':
          new_list.append('X'+subtune)
        else:
          new_list.append(subtune)
      self.num_subtunes = len(new_list)
      return new_list

class Tune(object):
    """Initialize with either an ABC string or a json-parsed dict read from
    the TheSession API.
    """
    def __init__(self, abc=None, json=None):
        self.header = None
        self.full_measure_duration = 0
        self.is_tune_with_full_measures = False
        self.measure_duration_in_play_order = []
        self.play_order = []
        self.tokens = []
        self.measures = []

        if abc is not None:
            self.parse_abc(abc)
        elif json is not None:
            self.parse_json(json)
        else:
            raise TypeError("must provide abc or json")



    @property
    def url(self):
        try:
            return "http://thesession.org/tunes/%d#setting%d" % (self.header['reference number'], self.header['setting'])
        except KeyError:
            return None

    @property
    def notes(self):
        return [t for t in self.tokens if isinstance(t, Note)]


    def parse_abc(self, abc):
        self.abc = abc
        header = []
        tune = []
        in_tune = False
        for line in abc.split('\n'):
            line = re.split(r'([^\\]|^)%', line)[0]
            line = line.strip()
            if line == '':
                continue
            if in_tune:
                tune.append(line)
            else:
                if line[0] in info_keys and line[1] == ':':
                    header.append(line)
                    if line[0] == 'K':
                        in_tune = True
                elif line[:2] == '+:':
                    header[-1] += ' ' + line[2:]

        self.parse_header(header)
        self.parse_tune(tune)

    def parse_json(self, json):
        self.header = {
            "reference number": json['tune'],
            "setting": json['setting'],
            "tune title": json['name'],
            "meter": json['meter'],
            "unit note length": "1/" + json['meter'].split('/')[1],
            "key": json['mode'],
        }
        self.parse_tune(json['abc'].split('\r\n'))

    def parse_header(self, header):
        h = {}
        for line in header:
            key = line[0]
            data = line[2:].strip()
            h[info_keys[key].name] = data
        self.header = h

        if 'reference number' in h:
          self.reference = h['reference number']
        if 'tune title' in h:
          self.title = h['tune title']
        self.key = h['key']
        self.full_measure_duration = self._get_correct_measure_length()


    @property
    def is_incomplete_only_one_measure(self):
      if self.is_tune_with_full_measures:
        return True
      if len(self.measures) > 0 and self.measure_duration_in_play_order[0] != self.full_measure_duration:
        for i in self.measure_duration_in_play_order[1:]:
          if i != self.full_measure_duration:
            return False
        return True
      return False

    def parse_tune(self, tune):

        self.tokens = self.tokenize(tune)
        self.measures = self.parse_measures(self.tokens)
        self._repair_measure_number()

        self._parse_offset()

        self.play_order, self.measure_duration_in_play_order = self._parse_play_order()
        self.is_tune_with_full_measures = self._check_measure_duration()
        self.is_ending_with_bar = self._check_last_is_bar()

        for measure in self.measures:
          for token in measure.tokens:
            token.meas_offset_from_repeat_start = measure.meas_offset_from_repeat_start

        meas_offset_missing = False
        note_measure_offset_missing = False
        for token in reversed(self.tokens):
          if token.meas_offset_from_repeat_start is None and not isinstance(token, Newline):
            print(f"Error in meas_offset_from_repeat_start: {token}")          
            meas_offset_missing = True
            break
        if meas_offset_missing:
          self.tokens = [token for token in self.tokens if token.meas_offset_from_repeat_start is not None]
        
        for token in (self.tokens):
          if token.measure_offset is None:
            print(f"Error in token measure_offset: {token}")
            note_measure_offset_missing = True
            break
        if note_measure_offset_missing:
          self.tokens = [token for token in self.tokens if token.measure_offset is not None]

    def parse_measures(self, tokens):
        """Parse the tokens into measures.
        """
        measures = []
        measure = None
        tokens_in_prev = []
        ignored_beam = []
        
        if isinstance(tokens[0], Beam):
          tokens[0].measure_number = 0
          tokens[0].measure_offset = 0

        for token in tokens:
            if isinstance(token, Beam):
                if len(tokens_in_prev) > 0:
                  # token.measure_number = len(measures)
                  tokens_in_prev.append(token)
                  measure = Measure(len(measures), tokens_in_prev, token._text, self.full_measure_duration)
                  # token.measure_offset = measure.duration
                  # TODO: handle measure with duration 0
                  # like measure with only line break '\n'
                  if measure.duration > 0:
                    if len(ignored_beam) == 1:
                      beam_text = ignored_beam[0]._text.replace('[1', '|1')
                      measure.prev_beam_text = beam_text
                      ignored_beam = []
                    elif len(ignored_beam) == 0:
                      if len(measures) > 0:
                        measure.prev_beam_text = measures[-1].text
                    else:
                      print("Error: so many ignored beam, ", ignored_beam)
                    measures.append(measure)
                  else:
                    # TODO: is it okay to ignore line break?
                    ignored_beam.append(token)
                  tokens_in_prev = []
                else:
                  token.measure_number = len(measures)
                  # TODO: It's not sure the below is safe
                  token.meas_offset_from_repeat_start = len(measures)
                  ignored_beam.append(token)
            else:
                tokens_in_prev.append(token)
        if len(tokens_in_prev) > 0:
          token.measure_number = len(measures)
          measure = Measure(len(measures), tokens_in_prev, token._text, self.full_measure_duration)
          token.measure_offset = measure.duration
          # TODO: handle measure with duration 0
          # like measure with only line break '\n'
          if measure.duration > 0:
            if len(ignored_beam) == 1:
              beam_text = ignored_beam[0]._text.replace('[1', '|1')
              measure.prev_beam_text = beam_text
              ignored_beam = []
            elif len(ignored_beam) == 0:
              if len(measures) > 0:
                measure.prev_beam_text = measures[-1].text
            else:
              print("Error: so many ignored beam, ", ignored_beam)
            measures.append(measure)
        return measures

    def _parse_play_order(self):
      '''

      This also update self.measures[i].meas_offset_from_repeat_start

      out: list of integer, list of float
      '''
      measure_text = [[measure.prev_beam_text, measure.text] for measure in self.measures]
      last_back_repeat_idx = 0
      measure_in_play_order = []
      measure_duration_in_play_order = [] 
      resolved_measure_ids = []
      resolved_first_repeat = []
      current_measure = 0
      cur_offset_from_repeat = 0
      while current_measure < len(measure_text):
        if current_measure in resolved_first_repeat:
          while ":|" not in measure_text[current_measure][1]:
            current_measure += 1
          current_measure += 1
          if current_measure >= len(measure_text):
            print("Error in play order: Repeat after first ending is strange")
            break
        measure_in_play_order.append(current_measure)
        self.measures[current_measure].meas_offset_from_repeat_start = cur_offset_from_repeat

        measure_duration_in_play_order.append(self.measures[current_measure].duration)
        meas = measure_text[current_measure]
        prev_beam, curr_beam = meas
        if "|:" in prev_beam:
          last_back_repeat_idx = current_measure
          cur_offset_from_repeat = 0
        if "|1" in prev_beam:
          resolved_first_repeat.append(current_measure)

        if ":|" in curr_beam  and current_measure not in resolved_measure_ids:
          resolved_measure_ids.append(current_measure)
          current_measure = last_back_repeat_idx
          cur_offset_from_repeat = 0
        else:
          if not self.measures[current_measure].is_incomplete or self.measures[current_measure].duration > self.full_measure_duration * 0.5:
            cur_offset_from_repeat += 1
          current_measure += 1
      return measure_in_play_order, measure_duration_in_play_order

    def _parse_offset(self):
        offset = 0
        for meas in self.measures:
          meas.tune_offset = offset
          offset += meas.duration
          for token in meas.tokens:
            token.tune_offset = meas.tune_offset + token.measure_offset

    def _get_correct_measure_length(self):
      header = self.header

      meter = header['meter']
      meter = meter.replace('C|', '2/2').replace('C', '4/4').replace(' 2/4', '2/4').replace('4/4|', '4/4')

      assert meter in ('2/2', '3/2', '4/4', '3/4', '2/4', '6/8', '9/8', '12/8'), 'Invalid meter: {}'.format(meter)
      meter = [int(x) for x in meter.split('/')]
      unit = header['unit note length']
      # unit = unit.replace('1/8', '1/8').replace('1/16', '1/16').replace('1/32', '1/32').replace('1/64', '1/64')
      assert unit in ('1/8', '1/16', '1/32', '1/64'), 'Invalid unit: {}'.format(unit)
      
      denom = int(unit.split('/')[1])
      return int(meter[0] / meter[1] * denom)

      if meter==[6,8] and unit=='1/8':
        return 6
      elif meter==[4,4] and unit=='1/8':
        return 8
      elif meter==[2,4] and unit=='1/16':
        return 8
      elif meter==[3,4] and unit=='1/8':
        return 6
      elif meter==[2,4] and unit=='1/8':
        return 4
      elif meter==[2,2] and unit=='1/8':
        return 8
      elif meter==[3,2] and unit=='1/8':
        return 12
      elif meter==[9,8] and unit=='1/8':
        return 9
      elif meter==[12,8] and unit=='1/8':
        return 12
      elif meter==[3,4] and unit=='1/16':
        return 12
      elif meter==[4,4] and unit=='1/16':
        return 16
      elif meter==[6,8] and unit=='1/16':
        return 12
      elif meter==[2,2] and unit=='1/16':
        return 16
      elif meter==[3,4] and unit=='1/32':
        return 24
      elif meter==[3,2] and unit=='1/16':
        return 24
      else:
        print(f"Not recognized meter and unit: meter {meter} and unit {unit}")
        # return 0
        raise NotImplementedError

    def _repair_measure_number(self):
      num_incomplete_meas = 0
      prev_lack = 0
      for measure in self.measures:
        if num_incomplete_meas > 0:
          measure.number -= num_incomplete_meas
          measure._update_meas_offset()
        if measure.is_incomplete:
          if prev_lack + measure.duration != measure.full_duration:
            num_incomplete_meas += 1
          prev_lack = measure.duration
          # else:
          #   prev_lack = 0
        # else:
        #   prev_lack = 0

    def _check_measure_duration(self):
      if len(self.measures) == 0:
        return False
      for i in range(len(self.measure_duration_in_play_order)-1):
        if self.measure_duration_in_play_order[i] != self.full_measure_duration  \
                and self.measure_duration_in_play_order[i] + self.measure_duration_in_play_order[i-1] != self.full_measure_duration \
                and self.measure_duration_in_play_order[i]+ self.measure_duration_in_play_order[i+1] != self.full_measure_duration:
          return False
      if len(self.measure_duration_in_play_order) == 1 and self.measure_duration_in_play_order[0] != self.full_measure_duration:
        return False
      if self.measures[-1].duration > self.full_measure_duration:
        return False
      return True

    def _check_last_is_bar(self):
      for i in range(1,10):
        last_token = self.tokens[-i]
        if '|' in  last_token._text:
          return True
        elif isinstance(last_token, Note) or isinstance(last_token, Rest):
          return False


    def tokenize(self, tune):
        # get initial key signature from header
        key = Key(self.header['key'])

        # get initial time signature from header
        meter = self.header.get('meter', 'free')
        unit = self.header.get('unit note length', None)
        # determine default unit note length from meter if possible
        if unit is None and meter != 'free':
            if eval(meter) < 0.75:
                unit = "1/16"
            else:
                unit = "1/8"
        tempo = self.header.get('tempo', None)
        time_sig = TimeSignature(meter, unit, tempo)


        tokens = []
        for i,line in enumerate(tune):
            # print(line)
            line = line.rstrip()

            if len(line) > 2 and line[1] == ':' and (line[0] == '+' or line[0] in tune_body_fields):
                tokens.append(BodyField(line=i, char=0, text=line))
                continue

            pending_dots = None
            j = 0
            while j < len(line):
                part = line[j:]
                # if part[:2] == "[1": # if repeat 1 is splitted from bar line by line break 
                #     print(part)

                # Field
                if part[0] == '[' and len(part) > 3 and part[2] == ':':
                    fields = ''.join(inline_fields.keys())
                    m = re.match(r'\[[%s]:([^\]]+)\]' % fields, part)
                    if m is not None:
                        if m.group()[1] == 'K':
                            key = Key(m.group()[3:-1])

                        tokens.append(InlineField(line=i, char=j, text=m.group()))
                        j += m.end()
                        continue

                # Space
                m = re.match(r'(\s+)', part)
                if m is not None:
                    tokens.append(Space(line=i, char=j, text=m.group()))
                    j += m.end()
                    continue

                # Note
                # Examples:  c  E'  _F2  ^^G,/4  =a,',3/2
                m = re.match(r"(?P<acc>\^|\^\^|=|_|__)?(?P<note>[a-gA-G])(?P<oct>[,']*)(?P<num>\d+)?(?P<slash>/+)?(?P<den>\d+)?", part)
                if m is not None:
                    g = m.groupdict()
                    octave = int(g['note'].islower())
                    if g['oct'] is not None:
                        octave -= g['oct'].count(",")
                        octave += g['oct'].count("'")

                    num = g.get('num', 1)
                    if g['den'] is not None:
                        denom = g['den']
                    elif g['slash'] is not None:
                        denom = 2 * g['slash'].count('/')
                    else:
                        denom = 1

                    tokens.append(Note(key=key, time=time_sig, note=g['note'], accidental=g['acc'],
                        octave=octave, num=num, denom=denom, line=i, char=j, text=m.group()))

                    if pending_dots is not None:
                        tokens[-1].dotify(pending_dots, 'right')
                        pending_dots = None

                    j += m.end()
                    continue

                # Beam  |   :|   |:   ||   
                m = re.match(r'([\|\:]+)([0-9])?', part)
                if m is not None:
                    if m.group() in '[]':
                        tokens.append(ChordBracket(line=i, char=j, text=m.group()))
                    else:
                        tokens.append(Beam(line=i, char=j, text=m.group()))
                    j += m.end()
                    continue

                # Chord  [ABC]
                # m = re.match(r'([\[\]])([0-9\-,])?', part)
                m = re.match(r'(?P<bracket>[\[\]])([0-9\-,])??(?P<num>\d+)?(?P<slash>/+)?(?P<den>\d+)?', part)
                if m is not None:
                    if m.group() in '[]':
                        tokens.append(ChordBracket(line=i, char=j, text=m.group()))
                    else:
                        g = m.groupdict()
                        num = g.get('num', 1)
                        if g['den'] is not None:
                            denom = g['den']
                        elif g['slash'] is not None:
                            denom = 2 * g['slash'].count('/')
                        else:
                            denom = 1
                        temp_back = 1
                        while temp_back < len(tokens) and not isinstance(tokens[-temp_back], ChordBracket):
                          if isinstance(tokens[-temp_back], (Note, Rest)):
                            tokens[-temp_back]._length = (num, denom)
                          temp_back += 1 
                        tokens.append(ChordBracket(line=i, char=j, text=m.group()))
                    # TODO: is it okay to delete this?
                    # else:
                    #     tokens.append(Beam(line=i, char=j, text=m.group()))
                    j += m.end()
                    continue

                # Broken rhythm
                if len(tokens) > 0 and (isinstance(tokens[-1], (Note, Rest)) or (isinstance(tokens[-1], ChordBracket) and tokens[-1]._text == ']')):
                    m = re.match('<+|>+', part)
                    if m is not None:
                        if isinstance(tokens[-1], (Note, Rest)):
                          tokens[-1].dotify(part, 'left')
                          pending_dots = part
                          j += m.end()
                          continue
                        else:
                          temp_back = 2
                          while temp_back < len(tokens) and isinstance(tokens[-temp_back], (Note, Rest)):
                            tokens[-temp_back].dotify(part, 'left')
                            temp_back += 1 
                          pending_dots = part
                          j += m.end()
                          continue

                # Rest
                m = re.match(r'([XZxz])(\d+)?(/(\d+)?)?', part)
                if m is not None:
                    g = m.groups()
                    tokens.append(Rest(g[0], num=g[1], denom=g[3], line=i, char=j, text=m.group()))

                    if pending_dots is not None:
                        tokens[-1].dotify(pending_dots, 'right')
                        pending_dots = None

                    j += m.end()
                    continue

                # Tuplets  (must parse before slur)
                m = re.match(r'\(([2-9])', part)
                if m is not None:
                    tokens.append(Tuplet(num=m.groups()[0], line=i, char=j, text=m.group()))
                    j += m.end()
                    continue

                # Slur
                if part[0] in '()':
                    tokens.append(Slur(line=i, char=j, text=part[0]))
                    j += 1
                    continue

                # Tie
                if part[0] == '-':
                    tokens.append(Tie(line=i, char=j, text=part[0]))
                    j += 1
                    continue

                # Embelishments
                m = re.match(r'(\{\/?\\?)|\}', part)
                if m is not None:
                    tokens.append(GracenoteBrace(line=i, char=j, text=m.group()))
                    j += m.end()
                    continue

                # Decorations (single character)
                if part[0] in '.~HLMOPSTuv':
                    tokens.append(Decoration(line=i, char=j, text=part[0]))
                    j += 1
                    continue

                # Decorations (!symbol!)
                m = re.match(r'\!([^\! ]+)\!', part)
                if m is not None:
                    tokens.append(Decoration(line=i, char=j, text=m.group()))
                    j += m.end()
                    continue

                # Continuation
                if j == len(line) - 1 and j == '\\':
                    tokens.append(Continuation(line=i, char=j, text='\\'))
                    j += 1
                    continue

                # Annotation
                m = re.match(r'"[\^\_\<\>\@][^"]+"', part)
                if m is not None:
                    tokens.append(Annotation(line=i, char=j, text=m.group()))
                    j += m.end()
                    continue

                # Chord symbol
                m = re.match(r'"[\w#/]+"', part)
                if m is not None:
                    tokens.append(ChordSymbol(line=i, char=j, text=m.group()))
                    j += m.end()
                    continue



                raise Exception("Unable to parse: %s\n%s" % (part, self.url))

            if not isinstance(tokens[-1], Continuation):
                tokens.append(Newline(line=i, char=j, text='\n'))

        return tokens

    def pitchogram(tune):
        hist = {}
        for note in tune.notes:
            v = note.pitch.abs_value
            hist[v] = hist.get(v, 0) + note.duration
        return hist


def get_thesession_tunes():
    import os, json
    if not os.path.isfile("tunes.json"):
        import sys, urllib
        url = 'https://raw.githubusercontent.com/adactio/TheSession-data/master/json/tunes.json'
        print("Downloading tunes database from %s..." % url)
        try:
            urllib.urlretrieve(url, 'tunes.json')
        except AttributeError:
            import urllib.request
            urllib.request.urlretrieve(url, 'tunes.json')
    return json.loads(open('tunes.json', 'rb').read().decode('utf8'))


if __name__ == '__main__':
    ts_tunes = get_thesession_tunes()
    for i,t in enumerate(ts_tunes):
        print("----- %d: %s -----" % (i, t['name']))
        tune = Tune(json=t)

    print("Header: %s" % tune.header)


    def show(tune):
        import pyqtgraph as pg
        plt = pg.plot()
        plt.addLine(y=0)
        plt.addLine(y=12)
        plt.addLine(x=0)

        ticks = []
        for i in (0, 1):
            for pitch in "CDEFGAB":

                ticks.append((i*12 + pitch_values[pitch], pitch))

        plt.getAxis('left').setTicks([ticks])

        tvals = []
        yvals = []

        t = 0
        for token in tune.tokens:
            if isinstance(token, Beam):
                plt.addLine(x=t)
            elif isinstance(token, Note):
                tvals.append(t)
                yvals.append(token.pitch.abs_value)
                t += token.duration
        plt.plot(tvals, yvals, pen=None, symbol='o')


        hist = tune.pitchogram()
        k = sorted(hist.keys())
        v = [hist[x] for x in k]
        plt = pg.plot()
        bar = pg.BarGraphItem(x=k, height=v, width=1)
        plt.addItem(bar)

        plt.getAxis('bottom').setTicks([ticks])
        