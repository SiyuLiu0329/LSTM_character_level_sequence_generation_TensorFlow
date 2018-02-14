

def read_file(path):
    txt = open('txt/dinos.txt', 'r').read()
    data = txt.lower()
    chars = list(set(data))
    return chars, data
