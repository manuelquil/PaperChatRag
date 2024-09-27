# PaperChatRag

## Instalation

- Install Ollama in your local. Download [Ollama](https://ollama.com/download) for your operative system.
- Install LLM models from the terminal (cmd in Win), 
  ````
  ollama pull llama3.1:8b
  ````
  ````
  ollama pull mistral
  ````
- Clone the repository in a local proyect folder. (in e.g. `C:\Users\%username%\Rag`).
-  Install the requirements files,
   ````
   pip install -r requirements.txt
   ````
- In console navigate to the proyect folder and execute main_app file,
   ````
   cd C:\Users\%username%\Rag
   python main_app.py
   ````
  ## Usage Example

### Upload a new Document and Create Knowledgment DataBase
This initial procedure must be done every time a new document is uploaded for chat. Once this is done, it is not necessary to repeat the procedure; just set the path where the Chroma database was written and start chatting.

  - Add a Path inside the Textbox to store the Knowledgment Document DataBase (e.g `c:\chroma`).
  - Upload a pdf Document and click Create Chroma DataBase (The app will create Chroma DataBase and automatically will store it into the Path.)
    

  
