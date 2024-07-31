# Q & A application with LangChain and OpenAI

This repository contains Python scripts for Q and A application using LangChain, an open-source library for natural language processing tasks.

## Overview

The scripts provided here perform various document processing tasks, including:

- Extracting text from PDF files
- Splitting text into smaller chunks
- Creating vector representations of document chunks
- Storing and retrieving vectors using Chroma, a vector store from LangChain
- Utilizing OpenAI Embeddings for text embedding tasks

## Installation

1. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

2. Set up environment variables by creating a `.env` file and adding necessary variables:

    ```
    OPENAI_API_KEY=value
    ```

## Usage

1. Ensure you have PDF files to process. Place them in the `process_files` directory.
2. Run the main script to start processing:

    ```bash
    streamlit run main.py
    ```

3. Follow the prompts to perform document processing tasks.

## Scripts

### `main.py`

This script serves as the entry point for document processing tasks. It performs the following operations:

- Extraction of text from PDF files
- Splitting of text into smaller chunks
- Creation of vector representations using LangChain's Chroma
- Storage and retrieval of vectors

### `process_files.py`

Contains functions for processing PDF files, including extraction of text and splitting into smaller chunks.

### `propmt.py`

Prompt Template for Retrieving Answers