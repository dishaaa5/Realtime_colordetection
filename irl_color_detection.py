import cv2
import numpy as np

web = cv2.VideoCapture(0)

while True:
    ret, img = web.read()
    if not ret:
        break

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # RED color (2 ranges because hue wraps at 180)
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    red_mask = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)

    # GREEN color
    lower_green = np.array([36, 100, 100])
    upper_green = np.array([86, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    # BLUE color
    lower_blue = np.array([94, 80, 2])
    upper_blue = np.array([126, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Combine masks with dilation
    kernel = np.ones((5, 5), np.uint8)

    for mask, color_name, box_color in [
        (red_mask, "RED", (0, 0, 255)),
        (green_mask, "GREEN", (0, 255, 0)),
        (blue_mask, "BLUE", (255, 0, 0))
    ]:
        mask = cv2.dilate(mask, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 300:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(img, (x, y), (x + w, y + h), box_color, 2)
                cv2.putText(img, f"{color_name} COLOR", (x, y), cv2.FONT_HERSHEY_PLAIN, 0.8, box_color, 2)

    cv2.imshow("Color Detection", img)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

web.release()
cv2.destroyAllWindows()
