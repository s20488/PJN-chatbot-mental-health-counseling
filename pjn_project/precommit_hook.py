import os
import subprocess


def sync_conda_with_poetry():
    # Список установленных через Conda пакетов
    conda_list = subprocess.check_output(["conda", "list"], text=True)

    # Проверка `pyproject.toml`
    with open(r"C:\Users\Anastasiia\Desktop\PJN\pjn_project\pyproject.toml", "r") as f:
        poetry_content = f.read()

    # Автоматическое добавление новых пакетов
    for line in conda_list.splitlines():
        if not line.startswith("#") and len(line.split()) >= 2:
            package = line.split()[0]
            if package not in poetry_content:
                os.system(f"poetry add {package}")

if __name__ == "__main__":
    sync_conda_with_poetry()
