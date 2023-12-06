from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores.chroma import Chroma
import os, shutil, time, pickle

app = Flask(__name__)
CORS(app)
DEVICE = "cpu"

# Uploads
app.config['UPLOAD_FOLDER'] = './static/files/unprocessed'

# Load embeddings
emModelSource = os.path.join(os.getcwd(), 'static', 'embeddings')
if not os.path.isdir(emModelSource):
    os.mkdir(emModelSource)
    emInstance = HuggingFaceInstructEmbeddings(
        model_name = "hkunlp/instructor-large",
        model_kwargs = {"device": DEVICE}
    )

    print(f"emInstance: {emInstance}")
    desPath = os.path.join(emModelSource, 'hkunlp-instructor-large.pkl')
    with open(desPath, 'wb') as f:
        pickle.dump(emInstance, f)

embeddingsPath = os.path.join(emModelSource, 'hkunlp-instructor-large.pkl')
with open(embeddingsPath, 'rb') as f:
    embeddings = pickle.load(f)

@app.route("/", methods=["GET"])
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/uploadFiles", methods=["GET", "POST"])
def saveFiles():
    if 'files' not in request.files:
        print(request.files['files'])
        print(request.files.items())
        print(dir(request.files))
        for i in request.files['files']:
            print(f"i: {i}")
        return jsonify("File part is missing")
    
    files = request.files.getlist('files')

    if not files:
        return jsonify("No files are selected")
    
    for file in files:
        file.save(app.config['UPLOAD_FOLDER'] + '/' + file.filename)

    return jsonify("success")

@app.route("/saveToDB", methods=["GET"])
def saveToDB():
    sourcePath = os.path.join(os.getcwd(), 'static', 'files', 'unprocessed')
    processed = os.path.join(os.getcwd(), 'static', 'files', 'processed')
    print(f"listDir: {os.listdir(sourcePath)}")
    loader = PyPDFDirectoryLoader(sourcePath)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=74)
    splitText = text_splitter.split_documents(docs)
    Chroma.from_documents(splitText, embeddings, persist_directory='chromaDB')

    for fileName in os.listdir(sourcePath):
        source = os.path.join(sourcePath, fileName)
        desPath = os.path.join(processed, fileName)
        if os.path.exists(desPath):
            os.remove(desPath)
        shutil.move(source, desPath)
        
    return jsonify("Files have been saved to the DB")

if __name__ == '__main__':
    app.run(debug=False, port=5001, host='0.0.0.0')