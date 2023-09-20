from flask import Flask, request, jsonify
import oci
import os
import base64
from flask_cors import CORS
from dotenv import load_dotenv
import json
import CAG_Process_Receipts
import CAG_Process_Exp_Notes

import pdf2image
import pandas as pd
import tempfile

load_dotenv('.env')

app = Flask(__name__)
CORS(app)

config = oci.config.from_file(os.getenv('CONFIG_PATH'), os.getenv('OCI_PROFILE'))
ai_vision_client = oci.ai_vision.AIServiceVisionClient(config, region="ap-mumbai-1")


# Helper function to check if a file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}


def analyze_image(image_file, year):
    year = int(year)
    pya = f"{year - 2}-{year - 1}"
    cya = f"{year - 1}-{year}"

    pya = str(pya)
    cya = str(cya)

    # image_binary = image_file.tobytes()
    # image_base64 = base64.b64encode(image_binary)
    # base64_string = image_base64.decode('utf-8')
    base64_string = image_file

    try:
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
    except Exception as e:
        return (str(e))

    result = str(analyze_image_response.data)  # The result from image analysis
    data = json.loads(result)
    df = CAG_Process_Receipts.process_json(data, pya, cya)

    return df


def process_image(image_data, year):
    final_df = CAG_Process_Exp_Notes.process_image_and_generate_dataframe(image_data, year)

    # Return the JSON response
    return final_df


@app.route('/test', methods=['GET'])
def test():
    return "Hello"


@app.route('/analyze_pdf', methods=['POST'])
def analyze_pdf():
    try:
        # Check if a file is included in the request
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})

        file = request.files['file']
        year = request.form['year']

        # Check if the file has a valid extension (e.g., .pdf)
        if not file.filename.endswith('.pdf'):
            return jsonify({'error': 'Invalid file format'})

        # Create a temporary directory to store images
        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_path = os.path.join(temp_dir, 'input.pdf')
            image_folder = os.path.join(temp_dir, 'images')
            os.makedirs(image_folder)

            # Save the uploaded PDF to a temporary file
            file.save(pdf_path)

            # Initialize lists to store binary image data
            images = pdf2image.convert_from_path(pdf_path)
            for i, image in enumerate(images):
                image.save(f'{image_folder}/x_page_{i + 1}.png', 'PNG')

            png_files = [file for file in os.listdir(image_folder) if file.endswith('.png')]

            # Initialize lists to store DataFrames
            analyze_df_list = []  # List for DataFrames from analyze_image
            process_df_list = []  # List for DataFrames from process_image

            # Process each image
            for i, image in enumerate(png_files):

                image_path = os.path.join(image_folder, image)
                # image = Image.open(image_path)
                # Analyze the image
                with open(image_path, "rb") as image_file:
                    image_data = image_file.read()
                    base64_image = base64.b64encode(image_data).decode('utf-8')

                analyze_result = analyze_image(base64_image, year)

                if isinstance(analyze_result, pd.DataFrame):
                    # Append the analyze DataFrame to the list
                    analyze_df_list.append(analyze_result)

                # Process the image if it's "exp_notes"
                elif analyze_result == "exp_notes":
                    with open(image_path, "rb") as image_file:
                        image_data = image_file.read()
                    processed_result = process_image(image_data, year)
                    if isinstance(processed_result, pd.DataFrame):
                        # Append the processed DataFrame to the list
                        process_df_list.append(processed_result)

                else:
                    continue

            # Combine DataFrames into separate DataFrames
            analyze_combined_df = pd.concat(analyze_df_list, ignore_index=True) if analyze_df_list else None
            process_combined_df = pd.concat(process_df_list, ignore_index=True) if process_df_list else None

            # Convert DataFrames to JSON
            analyze_json_data = analyze_combined_df.to_json(
                orient='records') if analyze_combined_df is not None else None
            process_json_data = process_combined_df.to_json(
                orient='records') if process_combined_df is not None else None

            # Create response JSON objects with 'type' field
            analyze_response = {'type': 'receipt', 'data': analyze_json_data}
            process_response = {'type': 'notes', 'data': process_json_data}

            return jsonify({'analyze_data': analyze_response, 'process_data': process_response})
    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run()
