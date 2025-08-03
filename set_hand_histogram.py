import cv2
import numpy as np
import pickle

# Load existing histogram if available
try:
    with open("hist", "rb") as f:
        hist = pickle.load(f)
except FileNotFoundError:
    hist = None

def build_squares(img):
    x, y, w, h = 420, 140, 10, 10
    d = 10
    imgCrop = None
    crop = None
    for i in range(10):
        for j in range(5):
            if imgCrop is None:
                imgCrop = img[y:y+h, x:x+w]
            else:
                imgCrop = np.hstack((imgCrop, img[y:y+h, x:x+w]))
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 1)
            x += w + d
        if crop is None:
            crop = imgCrop
        else:
            crop = np.vstack((crop, imgCrop))
        imgCrop = None
        x = 420
        y += h + d
    return crop

def get_hand_hist():
    # Try multiple camera indices to find a working one
    cam = None
    for i in range(5):
        cam_test = cv2.VideoCapture(i)
        if cam_test.isOpened():
            cam = cam_test
            print(f"Camera index {i} opened successfully")
            break
        else:
            cam_test.release()
    if cam is None:
        print("Error: Could not open any camera")
        return

    x, y, w, h = 300, 100, 300, 300
    flagPressedC, flagPressedS = False, False
    imgCrop = None
    hist_local = hist  # Use loaded hist if available

    while True:
        ret, img = cam.read()
        if not ret:
            print("Failed to grab frame")
            break
        img = cv2.flip(img, 1)
        img = cv2.resize(img, (640, 480))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        keypress = cv2.waitKey(1) & 0xFF

        if keypress == ord('c'):
            if imgCrop is not None:
                hsvCrop = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2HSV)
                hist_local = cv2.calcHist([hsvCrop], [0, 1], None, [180, 256], [0, 180, 0, 256])
                cv2.normalize(hist_local, hist_local, 0, 255, cv2.NORM_MINMAX)
                print("Histogram calculated and normalized")
                flagPressedC = True
            else:
                print("No image cropped yet to calculate histogram")

        elif keypress == ord('s'):
            if hist_local is not None:
                with open("hist", "wb") as f:
                    pickle.dump(hist_local, f)
                print("Histogram saved to 'hist' file")
            else:
                print("No histogram to save")
            flagPressedS = True
            break

        if flagPressedC:
            dst = cv2.calcBackProject([hsv], [0, 1], hist_local, [0, 180, 0, 256], 1)
            disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
            cv2.filter2D(dst, -1, disc, dst)
            blur = cv2.GaussianBlur(dst, (11, 11), 0)
            blur = cv2.medianBlur(blur, 15)
            ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            thresh = cv2.merge((thresh, thresh, thresh))
            cv2.imshow("Thresh", thresh)

        if not flagPressedS:
            imgCrop = build_squares(img)

        cv2.imshow("Set hand histogram", img)

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    get_hand_hist()
