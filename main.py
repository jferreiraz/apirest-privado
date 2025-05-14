from fastapi import FastAPI, UploadFile, File
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from fastapi.middleware.cors import CORSMiddleware
import torch
import os
import pdfplumber
import docx
import uuid
import gzip
from markdown_it import MarkdownIt
import re

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device in use: {device}")

model_name = "mistralai/Mistral-7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)

generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Message(BaseModel):
    message: str


@app.post("/chat")
def chat_with_bot(message: Message):
    prompt = f"""Você é um assistente inteligente e útil. Responda de forma educada e clara.
Usuário: {message.message}
Assistente:"""

    response = generator(
        prompt,
        max_new_tokens=100,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

    output_text = response[0]["generated_text"]
    resposta_limpa = output_text.split("Assistente:")[-1].strip()

    torch.cuda.empty_cache()

    return {"response": resposta_limpa}


UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


def extract_text_from_pdf(file_path):
    with pdfplumber.open(file_path) as pdf:
        text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
    return text


def extract_text_from_docx(file_path):
    doc = docx.Document(file_path)
    return "\n".join(p.text for p in doc.paragraphs)


UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
md = MarkdownIt()


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    # Verificando se o arquivo foi enviado corretamente
    if not file:
        return {"error": "Nenhum arquivo enviado."}

    original_filename = file.filename
    file_ext = original_filename.split(".")[-1].lower()

    # Sanitizando o nome do arquivo
    sanitized_filename = re.sub(r"[^a-zA-Z0-9.-]", "-", original_filename)  # Substitui caracteres inválidos por '-'

    # Verifica se o nome sanitizado é vazio
    if not sanitized_filename:
        return {"error": "Nome do arquivo inválido após sanitização."}

    file_path = os.path.join(UPLOAD_DIR, sanitized_filename)

    # Salvar o arquivo no disco
    try:
        # Verifica se o arquivo já existe
        if os.path.exists(file_path):
            return {"error": f"Um arquivo com o nome '{sanitized_filename}' já existe."}

        # Lê o conteúdo do arquivo e salva
        file_content = await file.read()
        print(f"Conteúdo do arquivo lido: {file_content[:100]}")  # Exibe os primeiros 100 bytes do conteúdo

        with open(file_path, "wb") as buffer:
            buffer.write(file_content)
        print(f"Arquivo salvo em: {file_path}")
    except Exception as e:
        return {"error": f"Erro ao salvar o arquivo: {str(e)}"}

    # Ler e extrair conteúdo
    try:
        text_content = ""
        if file_ext == "pdf":
            text_content = extract_text_from_pdf(file_path)
        elif file_ext == "docx":
            text_content = extract_text_from_docx(file_path)
        elif file_ext in ["md", "txt"]:
            with open(file_path, "r", encoding="utf-8") as f:
                text_content = f.read()
        else:
            return {"error": f"Tipo de arquivo '{file_ext}' não suportado."}

        print(f"Texto extraído: {text_content[:500]}")  # Exibe os primeiros 500 caracteres
    except Exception as e:
        return {"error": f"Erro ao processar o conteúdo do arquivo: {str(e)}"}

    # Converter o conteúdo extraído para Markdown
    try:
        markdown_text = md.render(text_content)
        print(f"Markdown gerado (início): {markdown_text[:300]}")
    except Exception as e:
        return {"error": f"Erro ao converter para Markdown: {str(e)}"}

    # Salva o Markdown compactado
    markdown_filename = f"{sanitized_filename}.md.gz"
    markdown_path = os.path.join(UPLOAD_DIR, markdown_filename)

    try:
        with gzip.open(markdown_path, "wt", encoding="utf-8") as gz_file:
            gz_file.write(markdown_text)
        print(f"Markdown compactado salvo em: {markdown_path}")
    except Exception as e:
        return {"error": f"Erro ao salvar Markdown compactado: {str(e)}"}

    return {"file_id": sanitized_filename, "original_filename": original_filename, "original_path": file_path, "markdown_path": markdown_path, "markdown_preview": markdown_text[:500]}  # Exibe só uma prévia


@app.get("/markdown/{file_id}")
async def get_markdown_file(file_id: str):
    try:
        # Localizando o arquivo .md.gz
        file_path = os.path.join(UPLOAD_DIR, f"{file_id}")
        if not file_path.endswith(".md.gz"):
            file_path += ".md.gz"
        print(file_path)

        if os.path.exists(file_path):
            # Descompactando o arquivo .gz
            with gzip.open(file_path, "rt", encoding="utf-8") as f:
                markdown_content = f.read()

            return {"markdown_content": markdown_content}  # Retornando o conteúdo como um objeto JSON
        else:
            return {"error": "Arquivo não encontrado."}
    except Exception as e:
        return {"error": f"Erro ao processar o arquivo: {str(e)}"}


@app.get("/files")
async def list_files():
    try:
        # Verificando se o diretório de uploads existe
        if not os.path.isdir(UPLOAD_DIR):
            return {"error": "Diretório de uploads não encontrado."}

        files = []
        # Listando os arquivos no diretório 'uploads'
        for filename in os.listdir(UPLOAD_DIR):
            file_path = os.path.join(UPLOAD_DIR, filename)
            if os.path.isfile(file_path):
                # Adicionando informações do arquivo à lista
                files.append(
                    {
                        "file_id": filename.split("_")[0],  # Extraímos o file_id da parte do nome do arquivo
                        "file_name": filename,
                        "file_size": os.path.getsize(file_path),
                        "file_path": file_path,
                        "file_ext": filename.split(".")[-1].lower(),
                    }
                )

        # Caso o diretório esteja vazio, retornamos uma resposta com mensagem
        if not files:
            return {"message": "Nenhum arquivo encontrado."}

        return {"files": files}

    except FileNotFoundError:
        return {"error": "Diretório de uploads não encontrado."}
    except PermissionError:
        return {"error": "Permissões insuficientes para acessar o diretório de uploads."}
    except Exception as e:
        return {"error": f"Erro ao listar os arquivos: {str(e)}"}
