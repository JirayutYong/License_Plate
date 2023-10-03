import numpy as np
from ultralytics import YOLO
import cv2
from sort.sort import *
from util import get_car, read_license_plate


# load models
coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO('./models/license_plate_detector.pt')
license_plate_recognition = YOLO('./models/license_plate_recognition.pt')

# save license plate
output_folder = './save_license'

# delete file before start
file_list = os.listdir(output_folder)
for file_name in file_list:
    file_path = os.path.join(output_folder, file_name)
    os.remove(file_path)

# test detector license
input_folder = './test_license'
input_files = os.listdir(input_folder)


for filename in input_files:
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        file_path = os.path.join(input_folder, filename)
        image = cv2.imread(file_path)


        # load video / images
        cap = cv2.VideoCapture(file_path)
        results = {}
        mot_tracker = Sort()
        vehicles = [2, 3, 5, 7]

        # read file in folder save license plate
        existing_files = os.listdir(output_folder)
        frame_number = max([int(filename.split('_')[1].split('.')[0]) for filename in existing_files], default=-1) + 1
        #frame_number = 0

        # read frames
        frame_nmr = -1
        ret = True
        while ret:
            frame_nmr += 1
            ret, frame = cap.read()
            if ret:
                results[frame_nmr] = {}
                # detect vehicles
                detections = coco_model(frame)[0]
                detections_ = []
                for detection in detections.boxes.data.tolist():
                    x1, y1, x2, y2, score, class_id = detection
                    if int(class_id) in vehicles:
                        detections_.append([x1, y1, x2, y2, score])

                # track vehicles
                track_ids = mot_tracker.update(np.asarray(detections_))

                # detect license plates
                license_plates = license_plate_detector(frame)[0]

                for license_plate in license_plates.boxes.data.tolist():
                    x1, y1, x2, y2, score, class_id = license_plate

                    # assign license plate to car
                    xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

                    if car_id != -1:
                        # crop license plate
                        license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

                        # Resize the license plate crop to your desired dimensions
                        desired_width = 300  # Desired width of the license plate
                        desired_height = 150  # Desired height of the license plate
                        license_plate_crop = cv2.resize(license_plate_crop, (desired_width, desired_height))

                        gray_license_plate = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                        mblur=cv2.medianBlur(gray_license_plate,5)
                        equalized_license_plate = cv2.equalizeHist(mblur)

                        gaussian_blur = cv2.GaussianBlur(gray_license_plate,(7,7),2)
                        sharp_plate = cv2.addWeighted(gray_license_plate,1.5,gaussian_blur,-0.5,0)

                        _, binary_license_plate = cv2.threshold(gray_license_plate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        contours, _ = cv2.findContours(binary_license_plate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


                        th1 = cv2.adaptiveThreshold(gray_license_plate,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,33,1)

                        if contours:
                            largest_contour = max(contours, key=cv2.contourArea)

                            # Approximate the contour to 4 points (assuming it's a rectangle)
                            epsilon = 0.05 * cv2.arcLength(largest_contour, True)
                            approximated_contour = cv2.approxPolyDP(largest_contour, epsilon, True)

                            # Draw the approximated contour on a blank image (black background)
                            contour_image = np.zeros_like(binary_license_plate)
                            cv2.drawContours(contour_image, [approximated_contour], -1, 255, thickness=cv2.FILLED)

                            # Find the orientation angle of the license plate
                            angle = cv2.minAreaRect(approximated_contour)[-1]
                            print(angle)

                            # Rotate the binary image to deskew the license plate
                            if angle < 360:
                                if angle > 100 or angle < -100:
                                    angle = 86
                                # Instead of fixed -90 degrees rotation, adjust the angle accordingly
                                rotated_angle = angle - 90
                                # Check if the angle is less than -45 after adjustment, and rotate accordingly
                                if rotated_angle < -15:
                                    rotated_angle = 0

                                angle = rotated_angle
                            print(angle)


                            rotation_matrix = cv2.getRotationMatrix2D(tuple(np.array(binary_license_plate.shape[1::-1]) / 2),
                                                                      angle, 1)
                            deskewed_license_plate = cv2.warpAffine(gray_license_plate, rotation_matrix,
                                                                    gray_license_plate.shape[1::-1], flags=cv2.INTER_LINEAR,
                                                                    borderMode=cv2.BORDER_CONSTANT)


                        # Define destination points (you can adjust these values as needed)
                        #dst_width = 300  # Desired width of the license plate
                        #dst_height = 150  # Desired height of the license plate
                        #dst_pts = np.array([[0, 0], [dst_width, 0],[0, dst_height],[dst_width, dst_height]], dtype=np.float32)

                        # Calculate the perspective transformation matrix
                        #matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

                        # Warp and transform the license plate
                        #warped_license_plate = cv2.warpPerspective(license_plate_crop, matrix, (dst_width, dst_height))

                        # Save the license plate image
                        filename = os.path.join(output_folder, f'frame_{frame_number}.png')
                        cv2.imwrite(filename,deskewed_license_plate)
                        frame_number += 1

                        # process license plate
                        #license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                        #_,license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)
                        recognition_output = license_plate_recognition.predict(source="save_license", conf=0.25,save=True)

                        # Display Output
                        #cv2.namedWindow('input', cv2.WINDOW_NORMAL)
                        #cv2.resizeWindow('input', 900, 600)
                        #cv2.imshow('input', image)
                        #cv2.imshow('crop', license_plate_crop)
                        #cv2.imshow('warp', deskewed_license_plate)
                        #cv2.waitKey(0)




