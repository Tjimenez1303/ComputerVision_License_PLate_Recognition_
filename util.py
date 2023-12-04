import string
import easyocr

# Initialize the OCR reader
reader = easyocr.Reader(['en'], gpu=False)

# Mapping dictionaries for character conversion
dict_char_to_int = {'O': '0',
                    'I': '1',
                    'J': '3',
                    'A': '4',
                    'G': '6',
                    'S': '5'}

dict_int_to_char = {'0': 'O',
                    '1': 'I',
                    '3': 'J',
                    '4': 'A',
                    '6': 'G',
                    '5': 'S'}

def get_car(license_plate, vehicle_track_ids):
    '''
    Recupera las coordenadas y el ID del vehículo en función de las coordenadas de la matrícula

    Args:
        license_plate (tuple): Tupla que contiene las coordenadas de la matrícula (x1, y1, x2, y2, puntuación, class_id).
        vehicle_track_ids (list): Lista de vehículos trackeados por ID y sus correspondientes coordenadas.

    Returns:
        tuple: Tupla que contiene las coordenadas del vehículo (x1, y1, x2, y2) y el ID.
    '''

    x1, y1, x2, y2, score, class_id = license_plate

    foundIt = False
    for j in range(len(vehicle_track_ids)):
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[j]

        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            car_indx = j
            foundIt = True
            break

    if foundIt:
        return vehicle_track_ids[car_indx]

    return -1, -1, -1, -1, -1

def read_license_plate(license_plate_crop):
    """
    Leer el texto de la matrícula desde la imagen recortada proporcionada.

    Args:
        license_plate_crop (PIL.Image.Image): Imagen recortada que contiene la matrícula.

    Returns:
        tuple: Tupla que contiene el texto de la matrícula formateado y su puntuación de confianza.
    """

    detections = reader.readtext(license_plate_crop)

    for detection in detections:
        bbox, text, score = detection

        text = text.upper().replace(' ', '')

        if license_complies_format(text):
            return format_license(text), score

    return None, None


def write_csv(results, output_path):
    """
    Write the results to a CSV file.

    Args:
        results (dict): Diccionario que contiene los resultados.
        output_path (str): Ruta al archivo CSV de salida.
    """
    with open(output_path, 'w') as f:
        f.write('{},{},{},{},{},{},{}\n'.format('frame_nmr', 'car_id', 'car_bbox',
                                                'license_plate_bbox', 'license_plate_bbox_score', 'license_number',
                                                'license_number_score'))

        for frame_nmr in results.keys():
            for car_id in results[frame_nmr].keys():
                print(results[frame_nmr][car_id])
                if 'car' in results[frame_nmr][car_id].keys() and \
                   'license_plate' in results[frame_nmr][car_id].keys() and \
                   'text' in results[frame_nmr][car_id]['license_plate'].keys():
                    f.write('{},{},{},{},{},{},{}\n'.format(frame_nmr,
                                                            car_id,
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][car_id]['car']['bbox'][0],
                                                                results[frame_nmr][car_id]['car']['bbox'][1],
                                                                results[frame_nmr][car_id]['car']['bbox'][2],
                                                                results[frame_nmr][car_id]['car']['bbox'][3]),
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][0],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][1],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][2],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][3]),
                                                            results[frame_nmr][car_id]['license_plate']['bbox_score'],
                                                            results[frame_nmr][car_id]['license_plate']['text'],
                                                            results[frame_nmr][car_id]['license_plate']['text_score'])
                            )
        f.close()

def license_complies_format(text):
    """
    Verificar si el texto de la placa de matrícula cumple con el formato requerido.

    Args:
        text (str): Texto de la placa de matrícula.

    Returns:
        bool: True si la placa de matrícula cumple con el formato, False en caso contrario.
    """
    if len(text) != 6:
        return False

    if (text[0] in string.ascii_uppercase or text[0] in dict_int_to_char.keys()) and \
       (text[1] in string.ascii_uppercase or text[1] in dict_int_to_char.keys()) and \
       (text[2] in string.ascii_uppercase or text[2] in dict_int_to_char.keys()) and \
       (text[3] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[3] in dict_char_to_int.keys()) and \
       (text[4] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[4] in dict_char_to_int.keys()) and \
       (text[5] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[5] in dict_char_to_int.keys()):
        return True
    else:
        return False
    
def format_license(text):
    """
    Formatear el texto de la placa de matrícula convirtiendo caracteres mediante diccionarios de mapeo.

    Args:
        text (str): Texto de la placa de matrícula.

    Returns:
        str: Texto de la placa de matrícula formateado.
    """
    license_plate_ = ''
    mapping = {0: dict_int_to_char, 1: dict_int_to_char, 2: dict_int_to_char,
               3: dict_char_to_int, 4: dict_char_to_int, 5:dict_char_to_int}
    for j in [0, 1, 2, 3, 4, 5]:
        if text[j] in mapping[j].keys():
            license_plate_ += mapping[j][text[j]]
        else:
            license_plate_ += text[j]

    return license_plate_