from flask import Flask,session, request, Response,jsonify
from flask_cors import CORS
import time
from dotenv import load_dotenv
import json
import os
import PyPDF2
import validators
from datetime import datetime
import faiss
import pickle
from embedding import text_embedding,query_embedding
from azure_blob_storage import get_data_from_blob,getEmbeddingFiles
from azure_files_share import getFiles,uploadFile_inazure,checkFileInAzure
import io
import numpy as np
#from pdf2image import convert_from_path


load_dotenv()
MAX_SIZE_MB = 2 * 1024 * 1024  # 5 MB in bytes

app = Flask(__name__)
app.secret_key = 'your_unique_secret_key123'  # Replace with a secure random string
CORS(app)


@app.route('/')
def index():
    return 'working'
###############################################################

@app.route('/chat', methods=['POST'])
def chat():
    # Try to get the user's message from the JSON payload
    user_message = request.json.get('message')
    file_name = request.json.get('fileName')
    indexfile = file_name + '.index'
    metadataFile = file_name + '.pkl'

    if not file_name:
        if not user_message:
            return jsonify({'error': 'Message is required.'}), 400

        # Create the prompt for the model
        prompt = "Please find the best answer to my question.\nQUESTION - " + user_message
        
        # Generate response from the model
        response = prompt_model.generate_content(prompt)
        
        if not response or not response.text:
            return jsonify({'error': 'Failed to generate a response.'}), 500

        return jsonify({'response': response.text})
    else:
        if not user_message:
            return jsonify({'error': 'Message is required.'}), 400

        # Embed the user message
        query_embed = query_embedding(user_message)
            # Retrieve the index from Azure Blob as bytes
        index_bytes = get_data_from_blob(indexfile, "vectorsfiles")
        # Deserialize the FAISS index from bytes
        #try:
        index_array = np.frombuffer(index_bytes, dtype=np.uint8)
        index = faiss.deserialize_index(index_array)
        #except Exception as e:
            #return jsonify({'error': f'Failed to load FAISS index: {str(e)}'}), 500

        # Load metadata (deserialize using pickle)
        time.sleep(2)

        metadata_bytes = get_data_from_blob(metadataFile, "metadafiles")
        #try:
        if isinstance(metadata_bytes, str):
            metadata_bytes = metadata_bytes.encode('utf-8')
        metadata = pickle.loads(metadata_bytes)  # Deserialize metadata from bytes
        #except Exception as e:
        #    return jsonify({'error': f'Failed to load metadata: {str(e)}'}), 500

        if metadata is None:
            return jsonify({'error': 'Metadata is empty or None.'}), 500

        # Perform search on the FAISS index
        k = 3  # Retrieve top 3 results
        try:
            distances, indices = index.search(np.array(query_embed).astype('float32'), k)
        except Exception as e:
            return jsonify({'error': f'Failed to search FAISS index: {str(e)}'}), 500

        # Retrieve the metadata for the nearest neighbors
        relevant_chunks = ''
        for idx in indices[0]:
            if idx != -1:  # Check if valid index
                relevant_chunks += metadata[idx]['text']

        # Create prompt for the model
        INSTRUCTION = "If the user asks for personal information such as patient name, license number, personal name, investigator officer name, or other sensitive information, respond with 'XYZ is the name or number, sorry, I do not provide personal information.'"
        makePrompt = f"PARAGRAPH - {relevant_chunks}\nUSER QUESTION - {user_message}\n{INSTRUCTION}"
        prompt = "Please find the best answer to the user's question from the given paragraph.\n" + makePrompt

        # Generate response from the model
        #response = prompt_model.generate_content(prompt)

        if not response or not response.text:
            return jsonify({'error': 'Failed to generate a response.'}), 500

        return jsonify({'response': response.text})



##############################################
@app.route('/files_get', methods=['GET'])
def getAllFiles():
    data = getFiles()
    return jsonify(data)
################################################
@app.route('/metadata_file', methods=['GET'])
def getMetadataFile():
    data = getEmbeddingFiles()
    return jsonify(data)

##############################################
def split_into_chunks(input_string, chunk_size=250):
    return [input_string[i:i + chunk_size] for i in range(0, len(input_string), chunk_size)]

def extract_text_from_pdf(file):
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ''
        for page in range(len(pdf_reader.pages)):
            text += pdf_reader.pages[page].extract_text()

        # Check if any text was extracted
        if text.strip():
            return text
        else:
            return 'ERROR'     
    except Exception as e:
        print(f"An error occurred: {e}")
##############################################
##############################################
# def extract_text_with_ocr(file_name):
#     try:
#         filename = os.path.splitext(file_name)[0]
#         images = convert_from_path(file_name)
#         pdftext=''
#         output = "ocr_img"
#         imgpath = []
#         os.makedirs(output,exist_ok=True)

#         for i , image in enumerate(images):
#             imagepath = os.path.join(output,f'page_{i+1}.png')
#             image.save(imagepath,"PNG")
#             imgpath.append(imagepath)

#         for image_path in imgpath:
#             results = reader.readtext(image_path)
#         # Print extracted text
#             for result in results:
#                 text = result[1]
#                 pdftext +=text

#         return pdftext

#     except Exception as e:
#         r = f"Error processing PDF {file_name}: {str(e)}"
#         return r    
    

##############################################
##############################################

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files.get('file')
    ocr_choice = request.form.get('ocr')  # Assuming OCR choice is passed in form data
    file_name = file.filename
    fname = os.path.splitext(file_name)[0]
    if not file:
        return jsonify({'error': 'No file provided'}), 400

    # Check if the file is a PDF
    if not file.filename.lower().endswith('.pdf'):
        return jsonify({'error': 'Only PDF files are allowed'}), 201

    uploaded_file_res = checkFileInAzure(file_name)  # Get existing files

    # Check if the file already exists
    if uploaded_file_res == 'EXISTS':
        return jsonify({'error': 'File already exists.'}), 202

    file_in_memory = io.BytesIO(file.read())

    if ocr_choice == 'true':
        # text = extract_text_with_ocr(file_in_memory)
        # chunks = split_into_chunks(text)
        # text_embedding(chunks,fname)
        # res = uploadFile_inazure(file)

        # for file in os.listdir('ocr_img'):
        #     if file.endswith('.png'):
        #         os.remove(os.path.join("ocr_img", file))
        # Perform OCR processing on the file (Add your OCR processing logic here)
        return jsonify({'message': f'File {file_name} OCR Not Support'}), 200
    else:
        # Process file without OCR
        text = extract_text_from_pdf(file_in_memory)
        if text == 'ERROR':
            return jsonify({'error': 'File already exists.'}), 203          
        #file_in_memory.close()
        chunks = split_into_chunks(text)

        text_embedding(chunks,fname)
        res = uploadFile_inazure(file) ## Upload file in azure file share

        return jsonify({'message': f'File {file_name} uploaded without OCR processing'}), 200

##############################################################
@app.route('/load_file', methods=['POST'])
def select_file():
    data = request.json
    file_name = data.get('fileName')

    filename = file_name+'.index'

    return jsonify({'message': file_name})




if __name__ == '__main__':
    app.run(debug=True)



