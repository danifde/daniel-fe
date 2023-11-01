import cv2
import face_recognition
import numpy as np
import os
import threading
import PySimpleGUI as sg

# Directorio donde se almacenan los vectores de embedding
embeddings_dir = 'embeddings'

# Directorio donde se almacenan las imágenes de referencia (personas conocidas)
known_faces_dir = 'faces'

# Inicializar el capturador de video para la cámara
video_capture = cv2.VideoCapture(0)

# Reducir la resolución de la cámara
video_capture.set(3, 640)  # Ancho
video_capture.set(4, 480)  # Alto

# Función para el registro biométrico
def register_user():
    ret, frame = video_capture.read()
    
    # Convertir el cuadro a escala de grises
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_locations = face_recognition.face_locations(gray_frame)
    
    if not face_locations:
        sg.Popup("No se detectó un rostro para el registro.")
        return
    
    # Tomar una foto del usuario en escala de grises para el registro
    face_image = gray_frame[face_locations[0][0]:face_locations[0][2], face_locations[0][3]:face_locations[0][1]]
    
    # Calcular los vectores de embedding
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    # Preguntar al usuario su nombre
    name = sg.PopupGetText("Por favor, ingresa tu nombre:")

    # Guardar la foto en blanco y negro en una carpeta
    if not os.path.exists(known_faces_dir):
        os.makedirs(known_faces_dir)
    cv2.imwrite(f"{known_faces_dir}/{name}.jpg", face_image)

    # Guardar los vectores de embedding en un archivo
    if not os.path.exists(embeddings_dir):
        os.makedirs(embeddings_dir)
    np.save(f"{embeddings_dir}/{name}.npy", face_encodings[0])

    sg.Popup(f"Usuario {name} registrado con éxito.")
    return name, face_image

# Función para el inicio de sesión biométrico
def authenticate_user():
    while True:
        ret, frame = video_capture.read()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_locations = face_recognition.face_locations(gray_frame)
        
        if face_locations:
            # Detectar y calcular los vectores de embedding del rostro detectado
            face_encodings = face_recognition.face_encodings(frame, face_locations)
            
            # Comparar con los usuarios registrados
            for file_name in os.listdir(embeddings_dir):
                registered_name = os.path.splitext(file_name)[0]
                registered_face_encodings = np.load(os.path.join(embeddings_dir, file_name))

                for encoding in face_encodings:
                    try:
                        results = face_recognition.compare_faces([registered_face_encodings], encoding)
                        if any(results):
                            name = sg.Popup(f"Bienvenido, {registered_name}. Acceso permitido.", non_blocking=True)
                            if name != sg.TIMEOUT_KEY:
                                display_face(frame, registered_name)
                                return
                    except Exception as e:
                        print("Error al comparar caras:", e)
            
            # Mostrar el cuadro de video en tiempo real
            for (top, right, bottom, left) in face_locations:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.imshow("Inicio Biométrico", frame)

        # Romper el bucle cuando se presione la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Función para mostrar el rostro y el nombre
def display_face(frame, name):
    cv2.putText(frame, name, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("Usuario Identificado", frame)

# Crear una interfaz gráfica con PySimpleGUI
layout = [
    [sg.Button("Registro Biométrico"), sg.Button("Inicio Biométrico")],
    [sg.Text("Haz clic en una opción para comenzar.")],
]

window = sg.Window("Sistema Biométrico", layout)

while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED:
        break
    if event == "Registro Biométrico":
        name, face_image = register_user()
        if name:
            display_face(face_image, name)
    if event == "Inicio Biométrico":
        authenticate_user()

window.close()

# Liberar recursos
video_capture.release()
cv2.destroyAllWindows()