def split_note(token):
  if '//' in token:
    pitch, duration = token.split('//')
    return ['pitch'+pitch, 'dur'+duration]
  else:
    return [token, '<pad>']