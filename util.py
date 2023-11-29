def get_car(license_plate, vehicule_track_ids):
    '''
    Recupera las coordenadas y el ID del vehículo en función de las coordenadas de la matrícula

    Args:
        license_plate (tuple): Tupla que contiene las coordenadas de la matrícula (x1, y1, x2, y2, puntuación, class_id).
        vehicle_track_ids (list): Lista de vehículos trackeados por ID y sus correspondientes coordenadas.

    Returns:
        tuple: Tupla que contiene las coordenadas del vehículo (x1, y1, x2, y2) y el ID.
    '''

    x1, y1, x2, y2, score, class_id = license_plate

    return 0,0,0,0,0