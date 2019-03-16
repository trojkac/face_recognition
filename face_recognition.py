import cv2
import numpy as np
from skimage import data
from skimage.filters import gaussian
from skimage.segmentation import active_contour
import scipy

# skin_filter_lower = np.array([0, 48, 80], dtype="uint8")
# skin_filter_upper = np.array([25, 255, 255], dtype="uint8")


skin_filter_lower = np.array([0, 51, 102], dtype="uint8")
skin_filter_upper = np.array([20, 153, 255], dtype="uint8")

# skin_filter_lower = np.array([6, 48, 60], dtype="uint8")
# skin_filter_upper = np.array([38, 153, 255], dtype="uint8")

skin_filter_lower2 = np.array([0, 51, 102], dtype="uint8")
skin_filter_upper2 = np.array([12, 155, 255], dtype="uint8")
skin_filter_lower2a = np.array([165, 51, 102], dtype="uint8")
skin_filter_upper2a = np.array([255, 155, 255], dtype="uint8")


def skin_detection(image):
    face_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    skin_mask = cv2.inRange(face_hsv, skin_filter_lower2, skin_filter_upper2)
    skin_mask2 = cv2.inRange(face_hsv, skin_filter_lower2a, skin_filter_upper2a)
    skin_mask = cv2.bitwise_or(skin_mask, skin_mask2)

    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    skin_mask = cv2.dilate(skin_mask, ker, iterations=4)
    skin_mask = cv2.erode(skin_mask, ker, iterations=4)

    skin_mask = cv2.GaussianBlur(skin_mask, (3, 3), 0)
    return skin_mask


def is_not_face(contour):
    x, y, w, h = cv2.boundingRect(contour)
    area = cv2.contourArea(contour)

    area_condition = area < 15000
    ratio_condition = w > 2 * h or w < h / 4
    return area_condition or ratio_condition


def is_eye(parentComponent, childComponent):
    pCont, pRect, pCent = parentComponent
    cCont, cRect, cCent = childComponent

    areaCondition = pRect[2] * pRect[3] * 0.015 < cRect[2] * cRect[3] and cRect[2] * cRect[3] < pRect[2] * pRect[
        3] * 0.25
    positionCondition = (pRect[1] + pRect[3] * 0.5) > cCent[1]
    return areaCondition and positionCondition


def pre_process(image):
    skin_mask = skin_detection(image)
    skin = cv2.bitwise_and(image, image, mask=skin_mask)

    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    face_gray = cv2.cvtColor(skin, cv2.COLOR_BGR2GRAY)
    face_gray = cv2.GaussianBlur(face_gray, (3, 3), 0)
    ker = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

    edges = cv2.Canny(face_gray, 100, 200)
    edges = cv2.dilate(edges, ker, iterations=10)

    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    hierarchy = hierarchy[0]
    face_contours = []
    face_hierarchy = []
    face_rectangles = []
    contour_mask = np.ones(skin.shape[:2], dtype="uint8") * 255
    for component in zip(contours, hierarchy):
        currentContour = component[0]
        currentHierarchy = component[1]
        x, y, w, h = cv2.boundingRect(currentContour)
        if currentHierarchy[3] < 0 and is_not_face(currentContour):
            cv2.drawContours(contour_mask, [currentContour], 0, 0, -1)
            continue

        face_contours.append(currentContour)
        face_hierarchy.append(currentHierarchy)
        face_rectangles.append((x, y, w, h))

    largest_contour_index = -1
    largest_contour_area = 0
    for rec, contour_i in zip(face_rectangles, range(0, len(face_rectangles))):
        x, y, w, h = rec
        if w * h > largest_contour_area:
            largest_contour_index = contour_i
            largest_contour_area = w * h
    moments = [cv2.moments(contour, False) for contour in face_contours]
    centers = []
    for moment in moments:
        centers.append((int(moment['m10'] / moment['m00']), int(moment['m01'] / moment['m00'])))

    face_components = zip(face_contours, face_rectangles, centers)
    main_face_component = face_components[largest_contour_index]

    eyes = []
    for component in face_components:
        if is_eye(main_face_component, component):
            eyes.append(component)

    eyes_mask = np.zeros(skin.shape[:2], dtype="uint8")
    for eye in eyes:
        cv2.drawContours(eyes_mask, [eye[0]], -1, 1, -1)
    eyes_image = cv2.bitwise_and(image_gray, image_gray, mask=eyes_mask)
    # eyes_image = cv2.equalizeHist(eyes_image)
    eyes_image = cv2.GaussianBlur(eyes_image, (3, 3), 0)
    eye_edges = cv2.Canny(eyes_image,150,200)
    #ret, eyes_image = cv2.threshold(eyes_image, 170, 255, cv2.THRESH_OTSU)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    eyes_image = cv2.dilate(eye_edges, ker, iterations=8)
    # eye_contours, eye_hierarchy = cv2.findContours(eyes_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # for contour in eye_contours:
    #     c, r = cv2.minEnclosingCircle(contour)
    #     cv2.circle(image, (int(c[0]), int(c[1])), int(r), (0, 0, 255), 3)
    #
    # cv2.drawContours(eyes_image, eye_contours, -1, 255, 3)
    cv2.imshow('bin', cv2.resize(eye_edges, (500, 500)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    x, y, w, h = main_face_component[1]
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)

    cv2.drawContours(image, face_contours, -1, (255, 0, 0), 2)
    for component in eyes:
        center = component[2]
        cv2.circle(image, center, 6, (0, 0, 255), 2, 8, 0)
        x, y, w, h = component[1]
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)

    # cv2.imshow('bin', cv2.resize(skin, (500, 500)))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return image
