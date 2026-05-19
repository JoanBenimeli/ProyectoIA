# 1. Usamos una imagen oficial de Python ligera
FROM python:3.11-slim

# 2. Le decimos a Docker en qué carpeta de su contenedor vamos a trabajar
WORKDIR /app

# 3. Copiamos el archivo de requisitos primero
COPY requirements.txt .

# 4. Instalamos las librerías necesarias
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copiamos todo el proyecto
COPY . .

# 6. Exponemos el puerto que usa Streamlit por defecto
EXPOSE 8501

# 7. Configuramos el comando que se ejecutará al arrancar el contenedor
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
