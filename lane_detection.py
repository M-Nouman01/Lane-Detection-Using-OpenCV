import cv2
import numpy as np

# Canny edge detection
def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny_image = cv2.Canny(blur, 50, 150)
    return canny_image

# Region of Interest (ROI)
def ROI(image):
    height = image.shape[0]
    width = image.shape[1]
    # Triangle for lane region
    triangle = np.array([[(200, height), (width-200, height), (width//2, int(height*0.4))]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, triangle, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

# Draw lines on image
def display_line(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            if line is not None:
                x1, y1, x2, y2 = line
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image

# Calculate coordinates from slope and intercept
def coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1 * 0.6)
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])

# Average left and right lane lines
def average_slope(image, lines):
    left_fit = []
    right_fit = []

    if lines is None:
        return []

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    left_line = coordinates(image, np.mean(left_fit, axis=0)) if len(left_fit) > 0 else None
    right_line = coordinates(image, np.mean(right_fit, axis=0)) if len(right_fit) > 0 else None

    return [left_line, right_line]

# Video processing
cap = cv2.VideoCapture("test2.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    canny_image = canny(frame)
    cropped_image = ROI(canny_image)
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    average_lines = average_slope(frame, lines)
    line_image = display_line(frame, average_lines)
    final_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)

    # Resize the final frame to 50% for smaller display
    scale_percent = 50
    width = int(final_image.shape[1] * scale_percent / 100)
    height = int(final_image.shape[0] * scale_percent / 100)
    resized_image = cv2.resize(final_image, (width, height), interpolation=cv2.INTER_AREA)

    # Show the resized video
    cv2.imshow("Result", resized_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
