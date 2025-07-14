all_omr_configs = {
    "config": {
        "0": {
            "template": "omr_config/JEE/template.json",
            "config": "omr_config/JEE/config.json",
            "reference_image_path": "omr_config/JEE/reference.jpg"
        },
        "1": {
            "template": "omr_config/NEET/template.json",
            "config": "omr_config/NEET/config.json",
            "reference_image_path": "omr_config/NEET/reference.jpeg",
            "hasAruco": True,
            "enableArucoAspectAngleValidation": True,
            "aruco_id_mappings_keys": [1,12,24,36],
            "aruco_aspect_ratio_threshold_error_rate": 7,
            "aruco_angle_threshold_error_rate": 5
        }
    }
}