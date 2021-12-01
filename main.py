import cv2
import dlib
import numpy as np
import time



# Funcion que extrae indices de un np array
def extract_index_nparray(nparray):
    index = None
    for num in nparray[0]:
        index = num
        break
    return index

def Capa(img):
    img = np.flip(img,axis=1)
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    #HSV values

    #setting the values for the cloak
    lower_red = np.array([130,35,56])
    upper_red = np.array([150,255,255])
#147-167
    mask1 = cv2.inRange(hsv, lower_red,upper_red)

    lower_red = np.array([170,35,56])
    upper_red =  np.array([190,255,255])
    mask2 = cv2.inRange(hsv,lower_red,upper_red)

    mask1 = mask1 +mask2

    mask1 = cv2.morphologyEx(mask1,cv2.MORPH_OPEN,np.ones((3,3),np.uint8), iterations = 2)
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_DILATE,np.ones((3,3),np.uint8), iterations = 1)

    mask2 =cv2.bitwise_not(mask1)

    res1 = cv2.bitwise_and(background,background,mask=mask1)
    res2 = cv2.bitwise_and(img,img,mask=mask2)
    final_output = cv2.addWeighted(res1,1,res2,1,0)
    return final_output


# Leemos imagen
img = cv2.imread("faces/bradley_cooper.jpg")
# Imagen en grises
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Máscara donde vamos a extraer la cara 1. Es una imagen fondo negro del mismo tamaño de la imagen original.
mask = np.zeros_like(img_gray)  # Máscara de un solo canal

# Captura de video
cap = cv2.VideoCapture(0)

time.sleep(3)       # parantheis haas two because the camera needs time to adjust it self i according to the environment(ANDHERA KAMRA)

background = 0


# Face detector de dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Faces que encontramos con detector en la imagen en grises
faces = detector(img_gray)  # Es una lista o lista de listas si hay más de una cara

# Recorremos todas las caras que pueda haber en la imagen
for face in faces: 
    landmarks = predictor(img_gray, face)  # landmarks de cada cara en escala de grises
    landmarks_points = []                  # Lista donde pondremos cada punto de landmark   
    for n in range(0, 68):                 # Cada landmark tiene 68 puntos
        # Cogemos la longitud en x del landmark, y luego la de y.
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        landmarks_points.append((x, y))
    
    # OpenCv necesitan los puntos como numpy array. Numpy va más rápido.
    points = np.array(landmarks_points, np.int32)
    
    # convexHull = parte externa de los landmarks = parte que queremos recortar
    # No permite angulos de mas de 180 grados, por eso no hace el triangulo ceja-nariz-ceja
    convexhull = cv2.convexHull(points)
    
    # Aqui ponemos en nuestra máscara (el fondo negro) el convexHull
    cv2.fillConvexPoly(mask, convexhull, 255)
    
    # Esto pone la cara de bradley en el hueco blanco
    face_image_1 = cv2.bitwise_and(img, img, mask=mask)
    
    # Delaunay triangulation
    rect = cv2.boundingRect(convexhull) # Esto encierra el convexHull en un rectangulo porque eso se traga Delauny
    subdiv = cv2.Subdiv2D(rect)         # Esto hace la triangulizacion en el el rectangulo que rodea a convexhull
    subdiv.insert(landmarks_points)     # Insertamos los puntos que quereos triangularizar
    triangles = subdiv.getTriangleList()
    triangles = np.array(triangles, dtype=np.int32) # Array de triangulicos
    
    # Recorremos cada triangulo
    indexes_triangles = []
    for t in triangles:
        pt1 = (t[0], t[1]) # Coordenadas del vertice 1
        pt2 = (t[2], t[3]) # Coordenadas del vertice 2
        pt3 = (t[4], t[5]) # Coordenadas del vertice 3
        
        # Encontramos los indices de los vértices
        index_pt1 = np.where((points == pt1).all(axis=1))
        index_pt1 = extract_index_nparray(index_pt1)
        
        index_pt2 = np.where((points == pt2).all(axis=1))
        index_pt2 = extract_index_nparray(index_pt2)
        
        index_pt3 = np.where((points == pt3).all(axis=1))
        index_pt3 = extract_index_nparray(index_pt3)
        
        # Si todo ok       
        if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
            triangle = [index_pt1, index_pt2, index_pt3] # Estos son los landmarks de los triangulos
            indexes_triangles.append(triangle)           # Los guardamos

            
# Capturing the background
for i in range(60):

    ret, background = cap.read()
    print(ret)
#capturing image            
background = np.flip(background,axis=1)            
        
# Vamos a buscar los triangulos de la segunda cara usando los landmarks
print ("POnte en la webcam")
time.sleep(3) 
while True:
    _, img2 = cap.read()
    img2 = Capa(img2)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img2_new_face = np.zeros_like(img2)
    faces2 = detector(img2_gray) # Detectamos cara de la segunda imagen
    for face in faces2:
        landmarks = predictor(img2_gray, face)
        landmarks_points2 = []
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            landmarks_points2.append((x, y))
            
        points2 = np.array(landmarks_points2, np.int32)
        convexhull2 = cv2.convexHull(points2)
        

    lines_space_mask = np.zeros_like(img_gray)
    lines_space_new_face = np.zeros_like(img2)
    # -------------------------------------------TRIANGUARIZACIÓN DE AMBAS CARAAS EN UN UNICO BUCLE-------------------------------
    for triangle_index in indexes_triangles:
        
        # PRIMERA CARA, triangulo 1
        tr1_pt1 = landmarks_points[triangle_index[0]]
        tr1_pt2 = landmarks_points[triangle_index[1]]
        tr1_pt3 = landmarks_points[triangle_index[2]]
        
        # Triangulo con sus 3 vertices
        triangle1 = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32)
        # Este es el rectangulo que rodea el triangulo
        rect1 = cv2.boundingRect(triangle1)
        (x, y, w, h) = rect1 # x e y del rectangulo, anchura y altura
        cropped_triangle = img[y: y + h, x: x + w]    # Este es el triangulo recortado del rectangulo que acabamos de crear
        cropped_tr1_mask = np.zeros((h, w), np.uint8) # Esta es una mascara de fondo negro del tamaño del triangulo.
        
        points = np.array([[tr1_pt1[0] - x, tr1_pt1[1] - y],
                        [tr1_pt2[0] - x, tr1_pt2[1] - y],
                        [tr1_pt3[0] - x, tr1_pt3[1] - y]], np.int32)
        cv2.fillConvexPoly(cropped_tr1_mask, points, 255) # Relleno la mascara con los puntos
    
    
        # TRIANGULATION SEGUNDA CARA, triangulo2. Mismo proceso
        tr2_pt1 = landmarks_points2[triangle_index[0]]
        tr2_pt2 = landmarks_points2[triangle_index[1]]
        tr2_pt3 = landmarks_points2[triangle_index[2]]
        
        triangle2 = np.array([tr2_pt1, tr2_pt2, tr2_pt3], np.int32)
        
        rect2 = cv2.boundingRect(triangle2)
        (x, y, w, h) = rect2
        #cropped_triangle2 = img2[y: y + h, x: x + w]
        cropped_tr2_mask = np.zeros((h, w), np.uint8)
        points2 = np.array([[tr2_pt1[0] - x, tr2_pt1[1] - y],
                            [tr2_pt2[0] - x, tr2_pt2[1] - y],
                            [tr2_pt3[0] - x, tr2_pt3[1] - y]], np.int32)
        cv2.fillConvexPoly(cropped_tr2_mask, points2, 255)
    
        # WARP: DEFORMACION DE LOS TRIANGULOS
        # Conversión a float 32
        points = np.float32(points)
        points2 = np.float32(points2)
        # Matrix de transformación (deformación)
        M = cv2.getAffineTransform(points, points2)
        # Cogemos el triangulo de la imagen 1 (cropped_triangle) y lo transformamos a la width and height del segundo
        warped_triangle = cv2.warpAffine(cropped_triangle, M, (w, h))
        warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=cropped_tr2_mask)

    
    
    # RECONSTRUCCION DE LA CARA DESTINO CON LOS TRIANGULOS DEFORMADITOS
        img2_new_face_rect_area = img2_new_face[y: y + h, x: x + w]
        img2_new_face_rect_area_gray = cv2.cvtColor(img2_new_face_rect_area, cv2.COLOR_BGR2GRAY)
        _, mask_triangles_designed = cv2.threshold(img2_new_face_rect_area_gray, 1, 255, cv2.THRESH_BINARY_INV)
        warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=mask_triangles_designed)

        img2_new_face_rect_area = cv2.add(img2_new_face_rect_area, warped_triangle)
        img2_new_face[y: y + h, x: x + w] = img2_new_face_rect_area # En la posicion especificada de la cara 2, plantamos el triangulo
        
    # Face swapped (putting 1st face into 2nd face)
    img2_face_mask = np.zeros_like(img2_gray)
    img2_head_mask = cv2.fillConvexPoly(img2_face_mask, convexhull2, 255)
    img2_face_mask = cv2.bitwise_not(img2_head_mask)



    img2_head_noface = cv2.bitwise_and(img2, img2, mask=img2_face_mask)
    result = cv2.add(img2_head_noface, img2_new_face)

    (x, y, w, h) = cv2.boundingRect(convexhull2)
    center_face2 = (int((x + x + w) / 2), int((y + y + h) / 2))

    #seamlessclone = cv2.seamlessClone(result, img2, img2_head_mask, center_face2, cv2.NORMAL_CLONE)
    seamlessclone = cv2.seamlessClone(result, img2, img2_head_mask, center_face2, cv2.MIXED_CLONE)

    cv2.imshow("img2", img2)
    cv2.imshow("clone", seamlessclone)
    cv2.imshow("result", result)



    key = cv2.waitKey(1)
    if key == 27:
        break


cap.release()
cv2.destroyAllWindows()