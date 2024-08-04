# Multi-document-MMRAG-gemini Application
It is a multimodal multi document RAG. Handles pdf, docx, xlsx, csv and image formats using gemini-flash for real-estate sector. 


## The MAIN app uses : 
1. streamlit
2. gemini-flash for convesation and gemini embeddings model through gemini api
3. Faiss for vector store
4. Langchain
5. tesseract-ocr
and other librarues mentioned in reuirements.txt 

## Contents
- [Introduction](#introduction)
- [The MAIN app uses](#the-main-app-uses)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Access the Application](#access-the-application)
  - [Additional Tips](#additional-tips)
- [System Architectural Flow](#system-architectural-flow)
- [Rough Work](#rough-work)
- [Tests Screenshots](#tests-screenshots)
- [Function Flowchart](#function-flowchart)

## Getting Started

Follow these steps to clone and run the application.

### Prerequisites

- Python 3.6 or higher
- `git` installed on your machine
- google's gemini api key

### Installation

1. **Clone the Repository**

    Open your terminal and run the following command:
    ```bash
    git clone https://github.com/ankitsrivastava637/multi-document-MMRAG-gemini.git
    ```

2. **Navigate to the Repository Directory**

    Change your current directory to the cloned repository:
    ```bash
    cd multi-document-MMRAG-gemini
    ```

3. **Set Up a Virtual Environment** (Optional but Recommended)

    Create a virtual environment to manage dependencies:
    ```bash
    python -m venv venv
    ```
    Activate the virtual environment:

    - On Windows:
      ```bash
      .\venv\Scripts\activate
      ```

    - On macOS and Linux:
      ```bash
      source venv/bin/activate
      ```

4. **Install Dependencies**

    Install the required dependencies using `pip`:
    ```bash
    pip install -r requirements.txt
    ```
5. **Set Google's gemini api key**    

6. **Run the Streamlit Application**

    Start the Streamlit application:
    ```bash
    streamlit run app.py
    ```
    This command will launch the Streamlit application in your default web browser. If the main script has a different name, replace `app.py` with the appropriate file name.

### Access the Application

Once the application is running, you can access it by opening the URL provided in the terminal (typically `http://localhost:8501`) in your web browser.

### Additional Tips

- If there are environment variables required by the application, make sure to set them up in your environment or create a `.env` file in the project directory.
- Check the repository's README file for any additional setup instructions or configuration options specific to the application.

By following these steps, you should be able to clone and run the Streamlit application successfully.


## SYSTEM ARCHITECTURAL FLOW :

![diagram-export-8-4-2024-1_17_22-PM](https://github.com/user-attachments/assets/c1bd389c-761b-46fc-a0f0-4d64fa2ee5ab)



## ROUGH WORK

Some Initial experimentation/rough work with huugingface open source models in google collab can be found in this : 
https://colab.research.google.com/drive/1IWITZ6Ye-CIfMLMM96V0OhPhJpRtiFqp?usp=sharing

- This collab has more was intended to design a more complex and customised RAG system with multi-vector retrieval and handling images, table in Pdfs, images through image descriptions using LLMs and further steps for better embedding and retrieval. 
- The idea in google collab rough rough was to experiment with multi-dcoument manual ingestion and running ETL pipeline for semi-structired and structured data of multiple complex documents and datasets - for retrieval. 
- The dataset/documents need more preprocessing and deep dynamic feature extraction steps to implemented for better retrieval. 
- Only the vector store and retrieval approach don't help. The quality of data processed and feature extracted also greatly affect the accuracy, quality and variety of retrieval. 
- The retreival will be as good as the processed data and features extracted.  



## Tests Screenshots : 
![image](https://github.com/user-attachments/assets/99d61911-748a-4be6-b52d-c21971e6030d)

![image](https://github.com/user-attachments/assets/07432d93-a4e5-49a9-91c7-e4ab43aa53ce)

![image](https://github.com/user-attachments/assets/1df8377a-7b9c-4474-a5fa-5a756c757d64)

![image](https://github.com/user-attachments/assets/6a656752-4e9c-4738-9a92-3bd3a11956a5)



## FUNCTION flowchart : 

![diagram-export-8-4-2024-12_56_04-PM](https://github.com/user-attachments/assets/338c941b-c7e1-42b2-bc47-49f40c364007)



