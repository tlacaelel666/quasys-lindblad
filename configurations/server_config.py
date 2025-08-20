# requirements.txt
"""
Flask==2.3.3
Flask-CORS==4.0.0
python-dotenv==1.0.0
gunicorn==21.2.0
pytest==7.4.2
requests==2.31.0
"""
# config.py - Configuración avanzada para diferentes entornos
import os
from datetime import timedelta

class Config:
    """Configuración base"""
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'quantum-dev-secret-key'
    JSON_SORT_KEYS = False
    JSONIFY_PRETTYPRINT_REGULAR = True
    
    # Configuración de la base de datos (cuando uses una real)
    DATABASE_URL = os.environ.get('DATABASE_URL') or 'sqlite:///quantumlink.db'
    
    # Configuración JWT (si implementas autenticación real)
    JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY') or SECRET_KEY
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=1)
    JWT_REFRESH_TOKEN_EXPIRES = timedelta(days=30)
    
    # Límites de rate limiting
    RATELIMIT_STORAGE_URL = os.environ.get('REDIS_URL') or 'memory://'
    
    # Configuración cuántica específica
    MAX_MESSAGE_LENGTH = int(os.environ.get('MAX_MESSAGE_LENGTH', '1000'))
    MAX_QUBITS_PER_MESSAGE = int(os.environ.get('MAX_QUBITS_PER_MESSAGE', '100'))
    QUANTUM_SIMULATION_TIMEOUT = int(os.environ.get('QUANTUM_SIMULATION_TIMEOUT', '30'))

class DevelopmentConfig(Config):
    """Configuración para desarrollo"""
    DEBUG = True
    TESTING = False
    
class ProductionConfig(Config):
    """Configuración para producción"""
    DEBUG = False
    TESTING = False
    
    # Configuración de logging para producción
    LOG_LEVEL = 'INFO'
    LOG_TO_STDOUT = os.environ.get('LOG_TO_STDOUT', 'true').lower() == 'true'

class TestingConfig(Config):
    """Configuración para testing"""
    TESTING = True
    DEBUG = True
    DATABASE_URL = 'sqlite:///:memory:'

# Mapeo de configuraciones
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}

# .env - Archivo de variables de entorno (crear en la raíz del proyecto)
"""
# Configuración de Flask
FLASK_APP=app.py
FLASK_ENV=development
FLASK_DEBUG=True

# Claves secretas (generar nuevas para producción)
SECRET_KEY=quantum-super-secret-key-change-in-production
JWT_SECRET_KEY=jwt-secret-key-change-in-production

# Base de datos
DATABASE_URL=sqlite:///quantumlink.db

# API Configuration
MAX_MESSAGE_LENGTH=1000
MAX_QUBITS_PER_MESSAGE=100
QUANTUM_SIMULATION_TIMEOUT=30

# Puerto del servidor
PORT=5000
"""

# docker-compose.yml - Para desarrollo con Docker
docker_compose_content = """
version: '3.8'

services:
  quantumlink-api:
    build: .
    ports:
      - "5000:5000"
    environment:
      - FLASK_ENV=development
      - FLASK_DEBUG=True
    volumes:
      - .:/app
    depends_on:
      - redis
      - postgres
    
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: quantumlink
      POSTGRES_USER: quantum
      POSTGRES_PASSWORD: quantum123
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
"""

# Dockerfile
dockerfile_content = """
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "app:app"]
"""