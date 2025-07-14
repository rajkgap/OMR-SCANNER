import cv2
import numpy as np
import time
from src.template import Template
from src.fetchOmrAnswers import process_omr
from src.utils.parsing import open_config_with_defaults
from src.logger import logger
from omr_config.omrs_configurations import all_omr_configs
from utils.math_utils import calculate_error_rates_of_aspect_and_angles, distance

TOTAL_POINTS = 8000
PERCENT_MATCH = 0.02
min_matches_length = TOTAL_POINTS*PERCENT_MATCH
nn_match_ratio = 0.8  # Nearest neighbor matching ratio
enableImageDebug = False

preloaded_omr_data = {}

def preload_omr_data():
    for omr_type, paths in all_omr_configs.get("config").items():
        preloaded_omr_data[omr_type] = paths
        config_path = paths["config"]
        template_path = paths["template"]
        reference_path = paths["reference_image_path"]

        # Load configurations, template, and reference image
        tuning_config = open_config_with_defaults(config_path)
        template = Template(template_path, tuning_config)
        reference_image = cv2.imread(reference_path, cv2.IMREAD_GRAYSCALE)
        # Extracts keypoints and descriptors from the reference image using the ORB algorithm.
        kp_ref, des_ref = orb.detectAndCompute(reference_image, None)

        if(paths.get("hasAruco", False)):
            corners, ids = detect_aruco_markers(reference_image)
            required_ids = paths.get("aruco_id_mappings_keys")
            missing_ids = set(required_ids).difference(ids)
            if missing_ids:
                raise ValueError(f"The reference image with omr_type {omr_type} don't have all required markers.")
            id_corner_mapping = {}

            for _id, corner in zip(ids, corners):
                id_corner_mapping[int(_id)] = corner[0]
            preloaded_omr_data[omr_type]["aruco_id_mappings"] = id_corner_mapping
            aruco_midpoints_dict = {key: np.mean(value, axis=0).tolist() for key, value in id_corner_mapping.items()}
            preloaded_omr_data[omr_type]["aruco_midpoints"] = aruco_midpoints_dict
            aruco_aspect_ratio = distance(aruco_midpoints_dict[required_ids[1]], aruco_midpoints_dict[required_ids[3]]) / distance(aruco_midpoints_dict[required_ids[0]], aruco_midpoints_dict[required_ids[1]])
            preloaded_omr_data[omr_type]["aruco_aspect_ratio"] = aruco_aspect_ratio


        preloaded_omr_data[omr_type]["template"] = template
        preloaded_omr_data[omr_type]["reference_image"] = reference_image
        preloaded_omr_data[omr_type]["kp_ref"] = kp_ref
        preloaded_omr_data[omr_type]["des_ref"] = des_ref
    logger.info("All OMR data preloaded.")


min_matches_length = int(TOTAL_POINTS * PERCENT_MATCH)

# Initializes the ORB (Oriented FAST and Rotated BRIEF) feature detector.
# TOTAL_POINTS is the maximum number of keypoints to detect.
orb = cv2.ORB_create(TOTAL_POINTS)

# You can use ORB for keypoint detection, and BEBLID for descriptor extraction (instead of ORB's default).
# This can increase robustness and accuracy.
beblid = cv2.xfeatures2d.BEBLID_create(0.75)

matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING)

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
parameters = cv2.aruco.DetectorParameters()  # Corrected attribute usage
def detect_aruco_markers(gray):
    # blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    # _, thresholded = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)
    corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    logger.info("Detected Aruco Ids: " + str(ids))
    logger.info("Rejected Aruco length: " + str(len(rejected)))
    if(ids is None or len(ids) == 0):
        return [], []
    print("ids is not none")
    return corners, ids.flatten()

def validate_omr_type(omr_type):
    if omr_type not in all_omr_configs.get("config"):
        raise ValueError(f"Invalid OMR Type: {omr_type}")

def increment_digits(s):
    # Iterate over each character in the string
    return ''.join(
        str((int(char) + 1) % 10) if char.isdigit() else char
        for char in s
    )

def final_method(img, omr_type):
    try:
        # Fetch preloaded template and reference image
        template = preloaded_omr_data[omr_type]["template"]
        # reference_image = preloaded_omr_data[omr_type]["reference_image"]
        st = time.time()
        in_omr = getAlignedImage(img, omr_type)
        logger.info(":::::::Started homographic image Deduction::::::::")
        detect_and_correct_curve_1(in_omr,1)
        logger.info(":::::::Completed homographic image Deduction::::::::")
        cv2.imwrite("raj_aligned_final_all_correct.jpg", in_omr)
        result = process_omr(in_omr, template)
        tt = time.time() - st
        logger.info("Total Time for processing OMR: " + str(tt))
        if "Roll_No" in result:
            result["Roll_No"] = increment_digits(result["Roll_No"])
        if "Booklet_No" in result:
            result["Booklet_No"] = increment_digits(result["Booklet_No"])
        return result
    except Exception as ex:
        logger.error("ERROR while processing final_method", ex)
        raise ex


def detect_and_correct_curve_1(image,val):
    blurred_temp = image
    if val == 0:
        blurred_temp = cv2.GaussianBlur(image, (5, 5), 0)

    # Use adaptive thresholding for better edge detection
    thresh = cv2.adaptiveThreshold(blurred_temp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # # Optional: Use morphological operations to close gaps in edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    blurred = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)


    ###Canny Method#########
    median = np.median(blurred)
    sigma = 0.33
    lower = int(max(0, (1.0 - sigma) * median))
    upper = int(min(255, (1.0 + sigma) * median))
    #lower sigma-->tighter threshold(default value of sigma can be 0.33)
    edge_image= cv2.Canny(blurred, lower, upper)
    if(enableImageDebug):
        cv2.imwrite("Canny_Image_A.jpg",edge_image)


    contours, hierarchy = cv2.findContours(edge_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours by area
    min_area = 10000
    filtered_contours = [c for c in contours if cv2.contourArea(c) > min_area]

    # Sort contours by area
    sorted_contours = sorted(filtered_contours, key=cv2.contourArea, reverse=True)
    i = 0
    global_aspect_ratio_deviation=0
    global_area_deviation=0
    global_edge_deviation=0
    rectangle_found = 0

    # Approximate and draw the largest rectangle
    for rectangular_contours in sorted_contours:
        largest_contour = rectangular_contours
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        if len(approx) == 4:  # Ensure it's a rectangle
            rectangle_found = 1
            cv2.drawContours(image, [approx], -1, (0, 255, 0), 3)
            if(enableImageDebug):
                if val == 1:
                    cv2.imwrite('ConvexHull_aligned.jpg', image)
                else:
                    cv2.imwrite('ConvexHull_not_aligned.jpg', image)
            print("Largest rectangle detected!")
            # Calculate deviations
            x, y, w, h = cv2.boundingRect(approx)
            bounding_area = w * h
            contour_area = cv2.contourArea(largest_contour)
            aspect_ratio = w / float(h)
            ideal_aspect_ratio = 1.0  # Assuming a perfect square as reference

            # Aspect Ratio Deviation
            aspect_ratio_deviation = abs(aspect_ratio - ideal_aspect_ratio)

            # Area Deviation
            area_deviation = abs(bounding_area - contour_area) / bounding_area

            # Edge Deviation
            edge_lengths = [np.linalg.norm(approx[i][0] - approx[(i + 1) % 4][0]) for i in range(4)]
            edge_deviation = max(edge_lengths) - min(edge_lengths)
            global_edge_deviation = max(global_edge_deviation,edge_deviation)
            global_area_deviation = max(global_area_deviation,area_deviation)
            global_aspect_ratio_deviation = max(global_aspect_ratio_deviation,aspect_ratio_deviation)
        i+=1
        if i==100:
            break;
    if rectangle_found == 0:
        print(f":::::: NOT ABLE TO FIND RECTANGLE ::::::")
    print(f":::MAX Aspect Ratio Deviation:::: {global_aspect_ratio_deviation:.2f}")
    print(f":::MAX Area Deviation:::: {global_area_deviation:.2%}")
    print(f":::MAX Edge Length Deviation:::: {global_edge_deviation:.2f} pixels")
    if global_area_deviation > 0.1025:
        raise ValueError("Place Paper On Plan Surface")

def detectAndTransForm(image, omr_config,height, width, borderVal):
    corners, ids = detect_aruco_markers(image)
    id_corner_mapping = {}
    for _id, corner in zip(ids, corners):
        id_corner_mapping[int(_id)] = corner[0]

    detected_coords = []
    destination_coords = []
    aruco_id_mappings = omr_config.get("aruco_id_mappings", {})
    for i in omr_config.get("aruco_id_mappings_keys", []):
        detected_coords.append(np.mean(id_corner_mapping[i], axis=0).tolist())
        destination_coords.append(np.mean(aruco_id_mappings[i], axis=0).tolist())

    print("dest_midpoints_dict: "+str(destination_coords))
    print("src_midpoints_dict: "+str(detected_coords))
    return warp_image(image, detected_coords, destination_coords, (width, height), borderVal)

def warp_image(image, src_points, dst_points, output_size, borderVal):
    src_points = np.array(src_points, dtype=np.float32)
    dst_points = np.array(dst_points, dtype=np.float32)
    if src_points.shape != (4, 2) or dst_points.shape != (4, 2):
        raise ValueError("Both src_points and dst_points must be 4x2 arrays.")
    if image is None:
        raise ValueError("Invalid image input.")

    M = cv2.getPerspectiveTransform(src_points, dst_points)
    return cv2.warpPerspective(image, M, output_size, borderValue=borderVal)

def validate_aruco(src_aruco_ids, ids, omr_config):
    logger.info("Validating aruco markers...")
    aruco_id_list = omr_config.get("aruco_id_mappings_keys")
    missing_ids = set(aruco_id_list).difference(set(ids))

    if missing_ids:
        logger.error("Missing aruco marker ids: ", missing_ids)
        raise ValueError(f"Image don't have all required markers.")

    aruco_midpoints_dict = {key: np.mean(value, axis=0).tolist() for key, value in src_aruco_ids.items()}
    print(aruco_midpoints_dict)
    aspect_threshold_error_rate = omr_config.get("aruco_aspect_ratio_threshold_error_rate")
    angle_threshold_error_rate = omr_config.get("aruco_angle_threshold_error_rate")
    p1, p2, p3, p4 = [aruco_midpoints_dict[i] for i in aruco_id_list]

    ideal_aspect_ratio = omr_config.get("aruco_aspect_ratio")
    response = calculate_error_rates_of_aspect_and_angles(p1, p2, p3, p4, ideal_aspect_ratio, 90)
    print("Threshold aspect error rate: ", aspect_threshold_error_rate)
    print("Threshold angle error rate: ", angle_threshold_error_rate)
    print("Error rates detected: ", response)
    if(response.get("error_aspect_ratio_1") > aspect_threshold_error_rate or response.get("error_aspect_ratio_2") > aspect_threshold_error_rate):
        raise ValueError(f"Align aruco markers correctly")
    for angle in response.get("error_angles"):
        if(angle > angle_threshold_error_rate):
            raise ValueError("Align aruco markers correctly")
    logger.info("Successfully validated aruco markers.")

def getArucoCoordinates(image, omr_config):
    # Detect markers
    logger.info("Detecting Aruco markers...")
    detected_coords = []
    destination_coords = []
    corners, ids = detect_aruco_markers(image)
    id_corner_mapping = {}
    for _id, corner in zip(ids, corners):
        id_corner_mapping[int(_id)] = corner[0]
    print("id_corner_mapping: "+str(id_corner_mapping))
    if(omr_config.get("enableArucoAspectAngleValidation", False)):
        validate_aruco(id_corner_mapping, ids, omr_config)

    aruco_id_mappings = omr_config.get("aruco_id_mappings", {})
    for i in omr_config.get("aruco_id_mappings_keys", []):
        for j in range(4):
            detected_coords.append(id_corner_mapping[i][j])
            destination_coords.append(aruco_id_mappings[i][j])
    detected_coords = np.array(detected_coords, dtype=np.float32)
    destination_coords = np.array(destination_coords, dtype=np.float32)
    logger.info("Detected all aruco markers of length: {}", len(detected_coords))
    return detected_coords, destination_coords

def remove_shadows(img):
    remove_start_time = time.time()
    dilated_img = cv2.dilate(img, np.ones((15, 15), np.uint8))
    bg_img = cv2.medianBlur(dilated_img, 45)
    diff_img = 255 - cv2.absdiff(img, bg_img)
    norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    logger.info("Time for removing shadows: " + str(time.time() - remove_start_time))
    return norm_img
def getAlignedImage(input_image, omr_type):
    start_time = time.time()
    logger.info("Inside getAlignedImage")

    try:
        # Validate OMR type
        if omr_type not in preloaded_omr_data:
            raise ValueError(f"OMR type '{omr_type}' is not preloaded.")

        # Load OMR configurations
        omr_config = preloaded_omr_data[omr_type]
        reference_image = omr_config["reference_image"]
        height, width = reference_image.shape[:2]
        median = np.median(input_image)
        median = (median, median, median)
        input_image = detectAndTransForm(input_image, omr_config,height, width, median)
        if(enableImageDebug):
            cv2.imwrite("Image_after_first_transform.jpg", input_image)
        detected_coords, destination_coords = [], []

        if(omr_config.get("hasAruco", False)):
            logger.info("Fetching aruco Coordinates from input image")
            detected_coords, destination_coords = getArucoCoordinates(input_image, omr_config)

        des_ref = omr_config["des_ref"]
        kp_ref = omr_config["kp_ref"]

        # Resize input image for faster processing (optional: adjust scale as needed)
        # input_image = cv2.resize(input_image, (0, 0), fx=0.5, fy=0.5)
        orb_start_time = time.time()
        kp_in, des_in = orb.detectAndCompute(input_image, None)
        logger.info("Time for input image detection with ORB: " + str(time.time() - orb_start_time))

        # Use k-nearest neighbors matching
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)  # Disable cross-checking for knnMatch
        bf_start_time = time.time()
        matches = bf.knnMatch(des_ref, des_in, k=2)
        logger.info("Time for BFMatcher: " + str(time.time() - bf_start_time))

        # Apply Lowe's ratio test
        good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
        logger.info("Total good matches: " + str(len(good_matches)))

        # if len(good_matches) < min_matches_length:
        #     logger.info(f"Not enough matches found: {len(good_matches)}. Minimum required is {min_matches_length}.")
        #     raise ValueError(f"The provided image is not a valid OMR image.")

        # Extract matched keypoints
        pts_ref = [kp_ref[match.queryIdx].pt for match in good_matches]
        pts_in = [kp_in[match.trainIdx].pt for match in good_matches]

        for sr, dt in zip(detected_coords, destination_coords):
            pts_in.append(sr)
            pts_ref.append(dt)
        pts_ref = np.float32(pts_ref)
        pts_in = np.float32(pts_in)
        # Find homography
        homography_start_time = time.time()
        H, mask = cv2.findHomography(pts_in, pts_ref, cv2.RANSAC, 5.0)
        logger.info("Time for homography: " + str(time.time() - homography_start_time))
        height, width = reference_image.shape[:2]
        wrap_start_time = time.time()

        aligned_image = cv2.warpPerspective(input_image, H, (width, height), borderValue=median)
        if(enableImageDebug):
            cv2.imwrite("second_aligned_homography.jpg", aligned_image)
        aligned_image = detectAndTransForm(aligned_image, omr_config,height, width, median)
        if(enableImageDebug):
            cv2.imwrite("third_aligned_homography.jpg", aligned_image)
        logger.info("Time for warpPerspective: " + str(time.time() - wrap_start_time))
        aligned_image = remove_shadows(aligned_image)
        if(enableImageDebug):
            cv2.imwrite("after_remove_shadow.jpg", aligned_image)
        ret, aligned_image = cv2.threshold(aligned_image, 156, 255, cv2.THRESH_BINARY)
        if(enableImageDebug):
            cv2.imwrite("after_threshold.jpg", aligned_image)
        logger.info("TOTAL TIME FOR ALIGNING IMAGE: " + str(time.time() - start_time))
        return aligned_image
    except Exception as ex:
        logger.error("ERROR while processing getAlignedImage", ex)
        raise ex