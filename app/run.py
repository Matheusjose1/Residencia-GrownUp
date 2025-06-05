# run.py
import subprocess
import sys
import os

def start_dev_server():
    """
    Inicia o servidor FastAPI usando Uvicorn com --reload.
    """
    print("Iniciando o servidor de desenvolvimento FastAPI (Uvicorn com --reload)...")

    # O comando que seria executado diretamente no terminal:
    # uvicorn app.main:app --reload

    # Usamos sys.executable para garantir que o interpretador Python do ambiente virtual seja usado,
    # e -m uvicorn para executar uvicorn como um módulo Python.
    command = [sys.executable, "-m", "uvicorn", "app.main:app", "--reload"]

    try:
        # subprocess.run executa o comando.
        # `check=True` fará com que uma CalledProcessError seja levantada se o comando retornar um código de saída diferente de zero.
        # `cwd=os.path.dirname(os.path.abspath(__file__))` garante que o comando é executado
        # a partir da raiz do projeto, que é onde `run.py` estará.
        subprocess.run(command, check=True, cwd=os.path.dirname(os.path.abspath(__file__)))
    except FileNotFoundError:
        print("\nErro: O comando 'uvicorn' não foi encontrado.")
        print("Certifique-se de que Uvicorn está instalado no seu ambiente virtual.")
        print("Você pode instalá-lo com: pip install uvicorn fastapi")
    except subprocess.CalledProcessError as e:
        print(f"\nErro ao executar o Uvicorn: {e}")
        print("Verifique se há erros no seu código FastAPI em 'app/main.py'.")
    except Exception as e:
        print(f"\nUm erro inesperado ocorreu: {e}")

if __name__ == "__main__":
    start_dev_server()