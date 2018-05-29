import os

def midi_files(dir):
    files = []
    for root, _, filenames in os.walk(dir):
        for f in filenames:
            if f.endswith('.mid'):
                files.append(root + f)
    return files
