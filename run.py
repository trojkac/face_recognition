import os

import cv2

import face_recognition as fr

INPUT_DIR = './faces/'
OUTPUT_DIR = './output/'


def process_all():
    for file_name in os.listdir(INPUT_DIR):
        process_image( file_name)


def process_image(file_name):
    print 'PROCESSING %s' % file_name
    try:
        face_image = cv2.imread(INPUT_DIR + file_name)
        skin_mask = fr.pre_process(face_image)
        #  cv2.imshow('skin', np.hstack([skin_mask]))
        cv2.imwrite(OUTPUT_DIR + file_name.split(".")[0] + '_s_de_edg_eye.jpg', skin_mask)
    except cv2.error as e:
        print 'ERROR - %s' % e.message


# process_image('z21670679Q.jpg')
process_all()
