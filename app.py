from flask import Flask, request, jsonify
import traceback
import logging

import cv2
import numpy as np

from service import validate_omr_type,final_method,preload_omr_data



app = Flask(__name__)
# It’s like saying: “Hey Flask, set up a web server for me.”
# When you run the script directly:
#
# python app.py
# Then inside that file:
#
# __name__ == "__main__"
# So:
#
# app = Flask(__name__)
# ... becomes:
#
# app = Flask("__main__")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# __name__ ensures logging is associated with this module.
# level=logging.INFO captures logs of INFO level and above.
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "OK"})



@app.route('/process_omr_upload', methods=['POST'])
def process_omr_upload():
    """
    Endpoint to process an OMR image sent in request body.
    omrType should be passed as a query param, e.g., /process_omr_upload?omrType=0
    """
    try:
        # Gets the value of omrType if present, otherwise returns "0" as default
        omr_type = request.args.get("omrType", "0")
        validate_omr_type(omr_type)

        # Get file from form-data
        if 'image' not in request.files:
            raise ValueError("No image file found in request")

        file = request.files['image']
        if file.filename == '':
            raise ValueError("Empty filename in uploaded image")

        # Read image from file
        image_bytes = file.read()
        image_np = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise ValueError("Could not decode image. Ensure it's a valid image format.")

        # Process image
        result = final_method(image, omr_type)
        return jsonify({"response": result, "status": 200}), 200

    except ValueError as ex:
        logger.error(f"Validation Error: {str(ex)}")
        return jsonify({"error": str(ex), "status": 400}), 400
    except Exception as ex:
        logger.debug(traceback.format_exc())
        return jsonify({"error": str(ex), "status": 500}), 500



if __name__ == "__main__":
    from waitress import serve
    logger.info("Preloading OMR configs and resources")
    preload_omr_data()
    logger.info("Completed preloading of OMR configs and resources")
    serve(app, host="0.0.0.0", port=8080)
