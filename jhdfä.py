import re

text = """

# Bahnhöfe: str(ID)
[Stations]
S1 5
S2 5

# Strecken: str(ID) str(Anfang) str(Ende) dec(Länge) int(Kapazität)
[Lines]
L1 S1 S2 1 3

# Züge: str(ID) str(Startbahnhof)/* dec(Geschwindigkeit) int(Kapazität)
[Trains]
T1 S1 1 1
T2 S1 1 1
T3 S1 1 1
T4 S1 1 1
T5 S1 1 1

# Passagiere: str(ID) str(Startbahnhof) str(Zielbahnhof) int(Gruppengröße) int(Ankunftszeit)
[Passengers]
P1 S1 S2 1 4
P2 S1 S2 1 4
P3 S1 S2 1 4

"""



print(trains)
# print(passengers)