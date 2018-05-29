import mido

def read_midi(filename):
  '''
    returns params, data with contents of filename.
    data has shape (-1, channel count).
  '''
  file = mido.MidiFile(filename)
  return extract_notes(file)


def extract_notes(midifile):
    notes = []

    for trk in midifile.tracks:
        note_stack = {}
        time = 0
        for msg in trk:
            time += msg.time
            note_on = msg.type == 'note_on' and msg.velocity > 0
            note_off = (msg.type == 'note_on' and msg.velocity == 0) or msg.type == 'note_off'
            if note_on:
                note_stack[msg.note] = (time, msg.velocity)
            elif note_off:
                n = note_stack.pop(msg.note, None)
                if n:
                    (t, v) = n
                    d = time-t
                    notes.append([msg.note])
    return notes

def write_notes(notes, filename):
    file = mido.MidiFile()
    track = mido.MidiTrack()
    file.tracks.append(track)

    for nt in notes:
        track.append(mido.Message('note_on', note=nt, velocity=100, time=0))
        track.append(mido.Message('note_off', note=nt, velocity=100, time=256))
    print track
    file.save(filename)
