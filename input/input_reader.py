import sys
input_file_path = sys.argv[1]

def read_input_file(String : input_file_path):
    f = open(input_file_path, 'r')
    text = f.read()
    
    trash = text.split('[Stations]')
    
    stations1 = trash[1].split('[Lines]')
    stations = str(stations1[0].replace('\n\n# Strecken: str(ID) str(Anfang) str(Ende) dec(LÃ¤nge) int(KapazitÃ¤t)\n', ''))[1:].strip()
    single_stations = stations.split('\n')
    station_numbers = len(single_stations)
    station_counter = 0
    station = {}
    for x in range(1, station_numbers+1):
        station["station%s" %x] = single_stations[station_counter]
        ++station_counter
    print(station)
    
    lines1 = stations1[1].split('[Trains]')
    lines = str(lines1[0].replace('\n', '').replace('# ZÃ¼ge: str(ID) str(Startbahnhof)/* dec(Geschwindigkeit) int(KapazitÃ¤t)', '')).strip()
    single_lines = lines.split('\n')
    lines_numbers = len(single_lines)
    line_counter = 0
    line = {}
    for x in range(1, lines_numbers+1):
        line["line%s" %x] = single_lines[line_counter]
        ++station_counter
    print(line)
    
    trains1 = lines1[1].split('[Passengers]')
    trains = str(trains1[0].replace('# Passagiere: str(ID) str(Startbahnhof) str(Zielbahnhof) int(GruppengrÃ¶ÃŸe) int(Ankunftszeit)', ''))[1:].strip()
    single_trains = trains.split('\n')
    trains_numbers = len(single_trains)
    train_counter = 0
    train = {}
    for x in range(1, trains_numbers+1):
        train["train%s" %x] = single_trains[train_counter]
        ++train_counter
    print(train)
    
    passengers = str(trains1[1])[1:].strip()
    single_passengers = passengers.split('\n')
    passengers_numbers = len(single_passengers)
    passenger_counter = 0
    passenger = {}
    for x in range(1, passengers_numbers+1):
        passenger["passenger%s" %x] = single_passengers[passenger_counter]
        ++passenger_counter
    print(passenger)
    
    return station, line, train, passenger
read_input_file(input_file_path)