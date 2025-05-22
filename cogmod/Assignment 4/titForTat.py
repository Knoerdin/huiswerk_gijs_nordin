#1 = collude 
#0 = defect

def play(roundNr,ownPrevMoves,otherPrevMoves,totalRounds):
  if roundNr == 0:
  # first move
    return 0
  if otherPrevMoves[roundNr-1]==1:
    return 1
  else:
    return 0
  return 0
  
def name():
  return 'titForTat'