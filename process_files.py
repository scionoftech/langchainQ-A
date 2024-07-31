import os
import shutil
import zipfile
from io import BytesIO
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

# Load variables from .env file into environment
load_dotenv(".env")

embedding = OpenAIEmbeddings(model="text-embedding-3-small")

persist_directory = './chroma'
folder_path = "process_files"


def check_persistent_path():
    """
        Check if the persistent directory exists.

        Returns:
            bool: True if the directory exists, False otherwise.
    """
    return os.path.exists(persist_directory)


# Function to extract zip file
def extract_zip(file):
    """
        Extract a zip file.

        Args:
            file (str): The zip file.

        Returns:
            bool: True if extraction is successful, False otherwise.
    """
    try:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"Folder '{folder_path}' created successfully.")
        zip_data = BytesIO(file.encode('utf-8'))
        with zipfile.ZipFile(zip_data, 'r') as zip_ref:
            zip_ref.extractall(folder_path)
    except Exception as e:
        print(e)
    return True


def delete_folder():
    """
        Delete the process_files and chroma folders.

        Returns:
            None
    """
    try:
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
        if os.path.exists(persist_directory):
            shutil.rmtree(persist_directory)
        print(
            f"Folder '{folder_path}' and its contents have been deleted successfully.")
    except OSError as e:
        print(f"Error: {folder_path} : {e.strerror}")


def list_pdf_files(directory):
    """
        List PDF files in the specified directory.

        Args:
            directory (str): The directory to search for PDF files.

        Returns:
            list: A list of paths to PDF files.
    """
    pdf_files = []
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.endswith(".pdf"):
                pdf_files.append(os.path.join(root, filename))
    return pdf_files


def process_files():
    """
        Process PDF files in the process_files directory.

        Returns:
            None
    """
    pdf_files = list_pdf_files(folder_path)

    loaders = [PyPDFLoader(pfile) for pfile in pdf_files]

    docs = []
    for loader in loaders:
        docs.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=150
    )

    # Create a split of the document using the text splitter
    splits = text_splitter.split_documents(docs)
    # print(splits)
    # Create the vector store
    _ = Chroma.from_documents(
        documents=splits,
        embedding=embedding,
        persist_directory=persist_directory
    )


# def get_similar_records(vectordb, question):
#     docs = vectordb.similarity_search(question, k=3)
#     print(len(docs))

#
def get_db():
    """
        Get the Chroma vector store.

        Returns:
            Chroma: The Chroma vector store.
    """
    vectordb = Chroma(persist_directory=persist_directory,
                      embedding_function=embedding)
    print(vectordb._collection.count())
    return vectordb


if __name__ == "__main__":
    process_files()
