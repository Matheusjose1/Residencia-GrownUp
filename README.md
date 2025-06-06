
# 🧠♻️Residência GrownUp: Análise de Resíduos com IA

Este repositório contém o projeto desenvolvido por alunos do 3º período do curso de Análise e Desenvolvimento de Sistemas da **Unit - Universidade Tiradentes**, como parte da fase **GrownUp** da residência tecnológica **Embarque Digital**, em parceria com a empresa **TPF Engenharia**. A aplicação utiliza **Inteligência Artificial**, especificamente o modelo YOLO (You Only Look Once), para realizar a análise e classificação de imagens de resíduos. O backend é construído utilizando o framework FastAPI em Python.

## 📌Visão Geral

O objetivo principal deste projeto é fornecer uma solução automatizada para a identificação e análise de diferentes tipos de resíduos a partir de imagens. A API permite o upload de imagens, que são então processadas por um modelo de detecção de objetos (YOLO) treinado para reconhecer categorias específicas de resíduos. Os resultados da análise, incluindo as classificações e possivelmente dados adicionais, são armazenados e podem ser consultados através de endpoints específicos.

Esta ferramenta pode ser útil em diversos contextos, como :
- Gestão de resíduos urbanos
- Reciclagem 
- Monitoramento ambiental
- Pesquisa acadêmica
Facilitando a coleta e análise de dados sobre a composição do lixo.

## 🚀 Funcionalidades Principais

- **Upload de Imagens:** Permite o envio de uma ou múltiplas imagens para análise.
- **Processamento com IA:** Utiliza o modelo YOLO para detectar e classificar resíduos nas imagens enviadas.
- **Armazenamento de Dados:** Salva as imagens processadas e os resultados da análise em um banco de dados (SQLite).
- **Consulta de Resultados:** Oferece endpoints para consultar os dados armazenados e os resultados das análises.
- **Interface Web (Implícita):** A presença de diretórios como `static` e o uso de `Jinja2Templates` sugerem a existência ou planejamento de uma interface web para interação com a API.

## 🛠️ Tecnologias Utilizadas

- **Backend:** Python  
- **Framework Web:** FastAPI  
- **Servidor ASGI:** Uvicorn  
- **Banco de Dados:** SQLite  
- **ORM:** SQLAlchemy  
- **Processamento de Imagem:** OpenCV, Pillow  
- **Detecção de Objetos (IA):** Ultralytics YOLO  
- **Manipulação de Dados:** NumPy  
- **Planilhas:** Openpyxl  
- **Upload de Arquivos:** python-multipart  

## 📁  Estrutura do Projeto

```

Residencia-GrownUp/
├── .vscode/                   # Configurações do VS Code
├── app/                       # Código principal da aplicação
│   ├── api/                   # Módulos da API (endpoints)
│   │   └── endpoints/         # Lógica dos endpoints
│   │       ├── db\_query\_router.py
│   │       └── image\_comparation.py
│   ├── core/                  # Configurações centrais e banco de dados
│   │   └── database.py
│   ├── crud/                  # Funções CRUD para o BD
│   │   └── crud\_image.py
│   ├── models/                # Modelos do banco de dados
│   │   └── image.py
│   ├── schemas/               # Validação de dados com Pydantic
│   │   └── image.py
│   └── main.py                # Ponto de entrada da aplicação
├── data/                      # Dados utilizados ou gerados
│   └── output/
│       └── imagens\_processadas/
├── static/                    # Arquivos estáticos da interface web
├── training/                  # Arquivos de treinamento do modelo YOLO
├── .gitattributes
├── .gitignore
├── app\_data.db                # Banco de dados SQLite
└── requirements.txt           # Dependências Python

````

## ⚙️ Instalação

1. **Clone o repositório:**
```bash
git clone https:https://github.com/Matheusjose1/Residencia-GrownUp.git
````

2. **Crie um ambiente virtual (recomendado):**

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

3. **Instale as dependências:**

```bash
pip install -r requirements.txt
```

## Uso

Para iniciar a API:

```bash
cd app
uvicorn main:app --reload
```

A API estará disponível em `http://127.0.0.1:8000`.
A documentação interativa (Swagger UI) pode ser acessada em `http://127.0.0.1:8000/docs`.

## 🔗  Endpoints Principais (inferidos)

* **Upload de Imagens:** `POST /images/upload`
* **Consulta de Resultados:** `GET /images/{image_id}` ou `GET /results`

## Integrantes – Squad 32

**(3º Período de ADS – Unit | Residência GrownUp – Embarque Digital | Empresa: TPF Engenharia)**

* Guilherme de Melo
* Mariana Freitas
* Matheus José
* Mateus Vinicius
* Murilo Vinicius
* Nicolas Soares
* Pedro Henrique
* Saulo Costa
