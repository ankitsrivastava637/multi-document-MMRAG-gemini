# Multi-document-MMRAG-gemini Application
It is a multimodal multi document RAG. Handles pdf, docx, xlsx, csv and image formats using gemini-flash for real-estate sector. 


## The MAIN app uses : 
1. streamlit
2. gemini-pro-flash for convesation and gemini embeddings model
3. Faiss for vector store
4. Langchain
5. tesseract-ocr
and other librarues mentioned in reuirements.txt 

Some Initial experimentation/rough work with huugingface open source models in google collab can be found in this : 
https://colab.research.google.com/drive/1IWITZ6Ye-CIfMLMM96V0OhPhJpRtiFqp?usp=sharing

- This collab has more was intended to design a more complex and customised RAG system with multi-vector retrieval and handling images, table in Pdfs, images through image descriptions using LLMs and further steps for better embedding and retrieval. 
- The idea in google collab rough rough was to experiment with multi-dcoument manual ingestion and running ETL pipeline for semi-structired and structured data of multiple complex documents and datasets - for retrieval. 



## Tests Screenshots : 
![image](https://github.com/user-attachments/assets/99d61911-748a-4be6-b52d-c21971e6030d)

![image](https://github.com/user-attachments/assets/07432d93-a4e5-49a9-91c7-e4ab43aa53ce)

![image](https://github.com/user-attachments/assets/1df8377a-7b9c-4474-a5fa-5a756c757d64)

![image](https://github.com/user-attachments/assets/6a656752-4e9c-4738-9a92-3bd3a11956a5)

