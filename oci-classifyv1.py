from flask import Flask, request, jsonify
import oci
import os
import base64
from flask_cors import CORS
from dotenv import load_dotenv
import CAG_Process_Exp_Notes, CAG_Process_Receipts
import pdf2image
import pandas as pd
import tempfile
import json
import re

load_dotenv('.env')

app = Flask(__name__)
CORS(app)

config = oci.config.from_file(os.getenv('CONFIG_PATH'), os.getenv('OCI_PROFILE'))
ai_vision_client = oci.ai_vision.AIServiceVisionClient(config, region="ap-mumbai-1")


def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def analyze_image(image_file, year):
    year = int(year)
    pya = f"{year - 2}-{year - 1}"
    cya = f"{year - 1}-{year}"

    pya = str(pya)
    cya = str(cya)

    try:
        analyze_image_response = ai_vision_client.analyze_image(
            analyze_image_details=oci.ai_vision.models.AnalyzeImageDetails(
                features=[
                    oci.ai_vision.models.ImageTextDetectionFeature(
                        feature_type="TEXT_DETECTION")
                ],
                image=oci.ai_vision.models.InlineImageDetails(
                    source="INLINE",
                    data=image_file)
            )
        )
    except Exception as e:
        return str(e)

    result = str(analyze_image_response.data)
    data = json.loads(result)
    df = CAG_Process_Receipts.process_json(data, pya, cya)

    return df


def process_image(image_data, year):
    return CAG_Process_Exp_Notes.process_image_and_generate_dataframe(image_data, year)


@app.route('/test', methods=['GE::q!T'])
def test():
    return "Hello"


@app.route('/analyze_pdf', methods=['POST'])
def analyze_pdf():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    year = request.form['year']
    file = request.files['file']

    if not file.filename.endswith('.pdf'):
        return jsonify({'error': 'Invalid file format'})

    with tempfile.TemporaryDirectory() as temp_dir:
        pdf_path = os.path.join(temp_dir, 'input.pdf')
        image_folder = os.path.join(temp_dir, 'images')
        os.makedirs(image_folder)

        file.save(pdf_path)

        images = pdf2image.convert_from_path(pdf_path)
        for i, image in enumerate(images):
            image.save(os.path.join(image_folder, f'page_{i + 1}.png'), 'PNG')

        analyze_df_list, process_df_list = [], []

        # Extract list of png files and sort them based on the page number
        png_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.png')],
                           key=lambda x: int(
                               x.split('_')[1].split('.png')[0]))  # This extracts the number from 'page_X.png'

        # Exclude the last image
        png_files = png_files[:-1]

        for image_name in png_files:
            if image_name.endswith('.png'):
                image_path = os.path.join(image_folder, image_name)

                with open(image_path, "rb") as image_file:
                    image_data = base64.b64encode(image_file.read()).decode('utf-8')

                    analyze_result = analyze_image(image_data, year)

                if isinstance(analyze_result, pd.DataFrame):
                    analyze_df_list.append(analyze_result)
                elif analyze_result == "exp_notes":
                    with open(image_path, "rb") as image_file:
                        image_data = image_file.read()
                    process_result = process_image(image_data, year)
                    if isinstance(process_result, pd.DataFrame):
                        process_df_list.append(process_result)

        analyze_combined_df = pd.concat(analyze_df_list, ignore_index=True) if analyze_df_list else None
        process_combined_df = pd.concat(process_df_list, ignore_index=True) if process_df_list else None

        # Check if analyze_combined_df is not None
        if analyze_combined_df is not None:
            analyze_data_content = {
                'data': json.loads(analyze_combined_df.to_json(orient='records')),
                'type': 'receipt'
            }
        else:
            analyze_data_content = None

        # Check if process_combined_df is not None
        if process_combined_df is not None:
            process_data_content = {
                'data': json.loads(process_combined_df.to_json(orient='records')),
                'type': 'notes'
            }
        else:
            process_data_content = None

        # Create the final dictionary
        output = {
            'analyze_data': analyze_data_content,
            'process_data': process_data_content
       }


        return json.dumps(output, indent=4)

    # with open('something.json', 'r') as file1:
    #     content = json.load(file1)
    # return content


@app.route('/analyze_pdf_test', methods=['POST'])
def analyze_pdf_test():
    uploaded_file = request.files['file']
    year = request.form['year']

    # Extract years using regular expression
    match = re.search(r'Fin(\d{4})(\d{4})Statement', uploaded_file.filename)

    if match:
        start_year = match.group(1)
        end_year = match.group(2)

        # Construct the expected year for the JSON file based on the provided year
        if int(year) >= 2018 and int(year) <= 2022:
            prev_year = int(year) - 1
            json_filename = f"Fin{prev_year}{year}.json"

            json_file_path = os.path.join("output_jsons", json_filename)
            if os.path.exists(json_file_path):
                with open(json_file_path, 'r') as file:
                    data = json.load(file)
                return jsonify(data), 200
            else:
                return jsonify({"error": "JSON file not found"}), 404
        else:
            return jsonify({"error": "Invalid year provided"}), 400
    else:
        return jsonify({"error": "Invalid file format"}), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, ssl_context=('/CAG_DocuAI/cert.pem', '/CAG_DocuAI/key.pem'))