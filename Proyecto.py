# Importamos librerías 
import sys
import cv2
from roboflow import Roboflow
from ultralytics import YOLO
from sort.sort import * # Librería para el seguimiento de objetos
import tensorflow as tf

#load models
#coco_model = YOLO("yolov8n.pt")

from util import get_car, read_license_plate, write_csv

# Configurar TensorFlow para usar la GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Actualmente, la memoria de la GPU se consume en su totalidad al inicio del proceso. 
        # Para permitir el crecimiento de la memoria de la GPU, se puede configurar como "crecimiento de la memoria".
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # El crecimiento de la memoria debe configurarse antes de que se inicialicen los dispositivos GPU
        print(e)

#!################################################# Cargamos el video ###############################################################
cap = cv2.VideoCapture('./sample5.mp4')
print("Video cargado")


#!################################################ Obtiene el Dataset y modelo de Roboflow ###########################################

# Key para acceder a la API de Roboflow que permitirá hacer solicitudes a la API de Roboflow
rf = Roboflow(api_key="Eyx3RPONOhgIHaM14u2O")

# Seleccionamos el proyecto de License Plate Recognition
project = rf.workspace("roboflow-universe-projects").project("license-plate-recognition-rxg4e")

# Descargamos la versión 4 del Dataset en formato YOLOV8
#dataset = project.version(4).download("yolov8")

# Obtiene el modelo asociado a la versión 4 del Dataset
model = project.version(4).model

#!################################################################### Pruebas ###########################################################

# '''
# Realizamos una predicción de una imagen de ejemplo y muestra el resultado en formato JSON.

#   - confindence: Cuán seguro está el modelo de que ha detectado un objeto en una ubicación específica de la imagen, en este caso, se filtran las predicciones
#   para incluir solo aquellas con un nivel de confianza del 40% o más.

#   - overlap: Cantidad de superposición entre dos áreas de detección. Se utiliza para manejar casos en los que el modelo detecta
#   el mismo objeto varias veces en ubicaciones ligeramente diferentes, en este caso, se filtra para solo incluir un nivel de superposición de 30% o menos.
# '''
# print(model.predict("Prueba_1.jpeg", confidence=40, overlap=30).json())

# # Observamos la predicción en una imagen llamada prediction.jpg
# model.predict("Prueba_1.jpeg", confidence=40, overlap=30).save("prediction.jpg")

# # Infer on an image hosted elsewhere
# model.predict("https://img-s-msn-com.akamaized.net/tenant/amp/entityid/AA11ViPy.img?w=4206&h=2804&m=4&q=80", hosted=True, confidence=5, overlap=70).save("prediction_2.jpg")

#!################################################################# Detección #########################################################

# Inicializamos el tracker de vehículos
mot_tracker = Sort()

# Llamamos a un modelo pre-entrenado de YOLOV8 para detectar vehículos
coco_model = YOLO('yolov8n.pt')

# Id de los vehiculos a detectar
vehicules = [2, 3, 5, 6, 7]       # car, motorbike, bus, train, truck

# Read frames
ret = True
frame_nmr = -1
results = {}


while ret:
  frame_nmr += 1
  ret, frame = cap.read()
  if ret:
        results[frame_nmr] = {} # Gurda cada fotograma de la imagen en un diccionario
        #?########################################### Detección de vehículos ################################
        detections = coco_model(frame)[0]
        detections_vehicules = []
        for detection in detections.boxes.data.tolist():
            '''
            - x1, y1, x2, y2: Coordenadas del cuadro delimitador
            - score: Valor de confidencia
            - class_id: Id de la clase (0: person, 1: bicycle ,2: car, etc.)
            '''
            x1, y1, x2, y2, score, class_id = detection
    
            # Si el vehiculo corresponde a los que vamos a detectar
            if int(class_id in vehicules):
                detections_vehicules.append([x1, y1, x2, y2, score])

        
        #?########################################### Seguimiento de vehículos ################################
        
        # Actualizamos el tracker de vehículos
        track_ids = mot_tracker.update(np.asarray(detections_vehicules))

        #?########################################### Detección de placas #####################################

        # Obtenemos las coordenadas de las placas
        detections_license_plates = model.predict(frame, confidence=40, overlap=30).json()
        # print(f"detections_license_plates es: {detections_license_plates}")

        # Observamos la predicción en una imagen llamada prediction.jpg
        cv2.imwrite("temp.jpg", frame)
        model.predict("temp.jpg", confidence=40, overlap=30).save("prediction.jpg")
        

        # Tupla para almacenar las coordenadas de las placas, la confidencia y el ID de la clase
        tuple_detection_license_plates = []

        for detection_license_plates in detections_license_plates['predictions']:
            x1, y1 = detection_license_plates['x'], detection_license_plates['y']
            width, height = detection_license_plates['width'], detection_license_plates['height']

            x1 = int(x1 - width/2)
            y1 = int(y1 - height/2) 
            x2, y2 = x1 + width, y1 + height
            score = detection_license_plates['confidence']
            class_id = detection_license_plates['class_id']

            tuple_detection_license_plates = (x1, y1, x2, y2, score, class_id)

            #?########################################### Asignación placa a carro #####################################
        
            # Obtenemos las coordenas del vehículo al que pertenece la placa
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(tuple_detection_license_plates,track_ids)

            #?########################################### Recorte placa de carro #####################################

            # print(f"x1: ${x1}, y1: ${y1}, x2: ${x2}, y2: ${y2}")

            if car_id != -1:
                # Recortamos la placa del vehículo
                license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :] # Frame 1-> y1:385 y2:410 x1:320 x2:395

                #?################################## Procesamiento de la placa recortada #####################################

                # Aplicamos escala de grises a la imagen
                license_plate_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)

                # Aplicamos thresholding a la imagen
                '''
                Thresholding se utiliza para convertir la imagen a una imagen binaria
            
                Params:
                    - 64: Valor de umbral. Todos los píxeles en la imagen con un valor menor que 64 se establecerán en 255, 
                    y todos los píxeles con un valor mayor o igual a 64 se establecerán en 0.
                    - 255: Valor de píxel máximo que se puede asignar a un píxel.
                    - cv2.THRESH_BINARY_INV: Umbralización binaria inversa. Los valores de intensidad por debajo del umbral 
                    se establecen en el valor máximo (255 en este caso) y los valores de intensidad por encima del umbral se establecen en 0.
                
                Returns:
                    - ret: Valor de umbral utilizado que no sirve para nada.
                    - license_plate_crop_thresh: Imagen binaria.
                '''
                _, license_plate_crop_thresh = cv2.threshold(license_plate_gray, 64, 255, cv2.THRESH_BINARY_INV)


                #?############################# Visualización de la placa recortada (PRUEBA) ################################
                # cv2.imshow('original_crop',license_plate_crop)
                # cv2.imshow('threshold',license_plate_crop_thresh)
                # cv2.waitKey(0)

                #! MINUTO 30:17

                #?############################# Leer el número de la placa ################################
                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

                if license_plate_text is not None:
                        results[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                                    'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                        'text': license_plate_text,
                                                                        'bbox_score': score,
                                                                        'text_score': license_plate_text_score}}

#?############################# Escribir resultados ################################
write_csv(results, './test.csv')