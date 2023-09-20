from flask import Flask, request, jsonify
import oci
import os
import base64
from flask_cors import CORS
from dotenv import load_dotenv
import json
from CAG_DocuAI import CAG_Process_Exp_Notes, CAG_Process_Receipts

load_dotenv('.env')

app = Flask(__name__)
CORS(app)

config = oci.config.from_file(os.getenv('CONFIG_PATH'), os.getenv('OCI_PROFILE'))
ai_vision_client = oci.ai_vision.AIServiceVisionClient(config, region="ap-mumbai-1")


# Helper function to check if a file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}


@app.route('/analyze_image_receipts', methods=['POST'])
def analyze_image():
    try:
        image_file = request.files['image']
        pya = request.form['prev_year']
        cya = request.form['curr_year']

        if not image_file:
            return jsonify({'error': 'Image file is required'}), 400

        image_binary = image_file.read()
        image_base64 = base64.b64encode(image_binary)
        base64_string = image_base64.decode('utf-8')

        analyze_image_response = ai_vision_client.analyze_image(
            analyze_image_details=oci.ai_vision.models.AnalyzeImageDetails(
                features=[
                    oci.ai_vision.models.ImageTextDetectionFeature(
                        feature_type="TEXT_DETECTION")
                ],
                image=oci.ai_vision.models.InlineImageDetails(
                    source="INLINE",
                    data=base64_string)
            )
        )

        result = str(analyze_image_response.data)  # The result from image analysis
        data = json.loads(result)
        df = CAG_Process_Receipts.process_json(data, pya, cya)
        json_data = df.to_json(orient='records')

        return json_data, 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/analyze_image_exp_notes', methods=['POST'])
def process_image():
    try:
        # Get the uploaded image from the request
        image_file = request.files['image']
        year = request.form['year']

        # Check if the image file exists and is of an allowed file type (e.g., PNG or JPEG)
        if image_file and allowed_file(image_file.filename):
            # Read the image from the request stream
            image_data = image_file.read()

            # Perform OCR and preprocessing on the in-memory image data
            final_df = CAG_Process_Exp_Notes.process_image_and_generate_dataframe(image_data, year)

            # Convert the final DataFrame to JSON
            final_json = final_df.to_json(orient='records')

            # Return the JSON response
            return final_json, 200

        else:
            return jsonify({'success': False, 'message': 'Invalid file format'})

    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})


@app.route('/test', methods=['GET'])
def test():
    return "Hello"


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
