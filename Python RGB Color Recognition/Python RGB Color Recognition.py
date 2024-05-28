import cv2
import numpy as np

# Kamera açma
cap = cv2.VideoCapture(0)

# Renk aralıkları
color_ranges = {
    'blue': (np.array([100, 50, 50]), np.array([130, 255, 255])),
    'red': (np.array([0, 50, 50]), np.array([10, 255, 255])),  # 0-10
    'green': (np.array([36, 25, 25]), np.array([86, 255, 255])),  # 36-86
    # Diğer renkler için gerekli renk aralıklarını ekleyebilirsiniz
}

while True:
    # Kamera görüntüsünü al
    ret, frame = cap.read()
    if not ret:
        break

    # Görüntüyü HSV formatına dönüştürme
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Her renk için nesne tanıma
    for color, (lower, upper) in color_ranges.items():
        mask = cv2.inRange(hsv, lower, upper)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 10000:  # Algılanacak minimum alanı artırın
                x, y, w, h = cv2.boundingRect(contour)

                # Her renk için uygun bir çerçeve rengi ve kalınlığı tanımlayın
                if color == 'blue':
                    frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)  # Mavi için kalın mavi çerçeve
                elif color == 'red':
                    frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255),
                                          1)  # Kırmızı için ince kırmızı çerçeve
                elif color == 'green':
                    frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0),
                                          2)  # Yeşil için orta kalınlıkta yeşil çerçeve

                # Her renk için uygun bir yazı rengi belirtin
                if color == 'blue':
                    text_color = (255, 0, 0)  # Mavi için mavi renk
                elif color == 'red':
                    text_color = (0, 0, 255)  # Kırmızı için kırmızı renk
                elif color == 'green':
                    text_color = (0, 255, 0)  # Yeşil için yeşil renk

                # Renk ismini belirtilen renkle yazdırın
                cv2.putText(frame, color.capitalize(), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)

    # Görüntüyü gösterme
    cv2.imshow('Object Detection', frame)

    # Çıkış için 'q' tuşuna bas
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kamerayı serbest bırak ve pencereyi kapat
cap.release()
cv2.destroyAllWindows()