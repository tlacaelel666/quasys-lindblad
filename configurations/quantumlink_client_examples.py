"""
Ejemplos de uso de la API QuantumLink
Incluye cliente Python, JavaScript y curl
"""

import requests
import json
from typing import Dict, Any, List

# ==================== CLIENTE PYTHON ====================

class QuantumLinkClient:
    """Cliente Python para la API QuantumLink"""
    
    def __init__(self, base_url: str = "http://localhost:5000", auth_token: str = None):
        self.base_url = base_url.rstrip('/')
        self.auth_token = auth_token
        self.session = requests.Session()
        
        if auth_token:
            self.session.headers.update({
                'Authorization': f'Bearer {auth_token  }'
            })
# 3. Decodificar un mensaje
"curl -X POST'http://localhost:5000/api/v1/decode' \
  "-H content-Type: application/json" \
  "-H Authorization: Bearer valid_user123" \
  -d",
     
    "package_id": "BiMO-12345678",
    "measurements": [
      {"energia_medida": 5.2, "timestamp": "2025-01-01T10:00:00Z"},
      {"energia_medida": 3.8, "timestamp": "2025-01-01T10:00:01Z"},
      {"energia_medida": 4.1, "timestamp": "2025-01-01T10:00:02Z"}
     ]

# 4. Obtener historial de paquetes
curl -X GET 'http://localhost:5000/api/v1/packages'\
  -H "Authorization: Bearer valid_user123"

# 5. Obtener detalles de un paquete específico
curl -X GET 'http://localhost:5000/api/v1/packages/BiMO-12345678' \
  -H "Authorization: Bearer valid_user123"

# 6. Ejemplo de error - sin autenticación
curl -X POST http://localhost:5000/api/v1/encode \
  -H "Content-Type: application/json" \
  -d '{"message": "Test without auth"}'
"""

# ==================== TESTS AUTOMATIZADOS ====================

def create_test_suite():
    """Crear suite de tests automatizados con pytest"""
    return '''
# test_quantumlink_api.py
import pytest
import json
from app import create_app

@pytest.fixture
def client():
    """Cliente de prueba para Flask"""
    app = create_app()
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

@pytest.fixture
def auth_headers():
    """Headers con autenticación válida"""
    return {
        'Authorization': 'Bearer valid_test_user',
        'Content-Type': 'application/json'
    }

class TestHealthCheck:
    """Tests para el endpoint de health check"""
    
    def test_health_check(self, client):
        """Test básico de health check"""
        response = client.get('/')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'success'
        assert 'QuantumLink API' in data['message']

class TestAuthentication:
    """Tests para autenticación"""
    
    def test_encode_without_auth(self, client):
        """Test de codificación sin autenticación"""
        response = client.post('/api/v1/encode', 
                             json={'message': 'test'},
                             content_type='application/json')
        assert response.status_code == 401
        data = json.loads(response.data)
        assert data['code'] == 'UNAUTHORIZED'
    
    def test_encode_with_invalid_token(self, client):
        """Test con token inválido"""
        headers = {'Authorization': 'Bearer invalid_token'}
        response = client.post('/api/v1/encode',
                             json={'message': 'test'},
                             headers=headers)
        assert response.status_code == 401
        data = json.loads(response.data)
        assert data['code'] == 'INVALID_TOKEN'

class TestEncoding:
    """Tests para codificación de mensajes"""
    
    def test_encode_valid_message(self, client, auth_headers):
        """Test de codificación exitosa"""
        message_data = {
            'message': 'QUANTUM TEST MESSAGE',
            'options': {
                'encoding_type': 'BiMO',
                'priority': 'high'
            }
        }
        response = client.post('/api/v1/encode',
                             json=message_data,
                             headers=auth_headers)
        assert response.status_code == 201
        data = json.loads(response.data)
        assert data['status'] == 'success'
        assert 'package_id' in data['data']
        assert data['data']['encoding_type'] == 'BiMO'
    
    def test_encode_empty_message(self, client, auth_headers):
        """Test con mensaje vacío"""
        response = client.post('/api/v1/encode',
                             json={'message': ''},
                             headers=auth_headers)
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['code'] == 'VALIDATION_ERROR'
    
    def test_encode_long_message(self, client, auth_headers):
        """Test con mensaje muy largo"""
        long_message = 'A' * 1001  # Excede el límite de 1000 caracteres
        response = client.post('/api/v1/encode',
                             json={'message': long_message},
                             headers=auth_headers)
        assert response.status_code == 400

class TestDecoding:
    """Tests para decodificación de mensajes"""
    
    def test_decode_nonexistent_package(self, client, auth_headers):
        """Test de decodificación de paquete inexistente"""
        decode_data = {
            'package_id': 'BiMO-nonexistent',
            'measurements': [{'energia_medida': 5.0}]
        }
        response = client.post('/api/v1/decode',
                             json=decode_data,
                             headers=auth_headers)
        assert response.status_code == 404
        data = json.loads(response.data)
        assert data['code'] == 'PACKAGE_NOT_FOUND'
    
    def test_decode_missing_measurements(self, client, auth_headers):
        """Test sin mediciones"""
        response = client.post('/api/v1/decode',
                             json={'package_id': 'BiMO-12345'},
                             headers=auth_headers)
        assert response.status_code == 400

class TestPackageManagement:
    """Tests para gestión de paquetes"""
    
    def test_get_packages_empty(self, client, auth_headers):
        """Test de historial vacío"""
        response = client.get('/api/v1/packages', headers=auth_headers)
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['data']['total_count'] == 0
    
    def test_get_package_details_not_found(self, client, auth_headers):
        """Test de detalles de paquete no encontrado"""
        response = client.get('/api/v1/packages/nonexistent', 
                            headers=auth_headers)
        assert response.status_code == 404

# Comando para ejecutar los tests:
# pytest test_quantumlink_api.py -v --tb=short
'''

# ==================== DOCUMENTACIÓN DE LA API ====================

api_documentation = """
# Documentación de la API QuantumLink

## Resumen
La API QuantumLink permite codificar y decodificar mensajes usando simulación cuántica BiMO.

## Base URL
```
http://localhost:5000
```

## Autenticación
Todas las rutas protegidas requieren un token de autorización en el header:
```
Authorization: Bearer {token}
```

Para pruebas, usa: `valid_user123`

## Endpoints

### GET /
Verificación de salud de la API.

**Respuesta:**
```json
{
  "status": "success",
  "message": "QuantumLink API está funcionando",
  "version": "1.0.0",
  "timestamp": "2025-01-01T12:00:00Z"
}
```

### POST /api/v1/encode
Codifica un mensaje cuántico.

**Headers requeridos:**
- `Authorization: Bearer {token}`
- `Content-Type: application/json`

**Body:**
```json
{
  "message": "MENSAJE A CODIFICAR",
  "options": {
    "encoding_type": "BiMO",
    "priority": "high|medium|low"
  }
}
```

**Respuesta exitosa (201):**
```json
{
  "status": "success",
  "data": {
    "package_id": "BiMO-12345678",
    "timestamp": "2025-01-01T12:00:00Z",
    "qubits_count": 18,
    "encoding_type": "BiMO"
  },
  "message": "Mensaje codificado exitosamente"
}
```

### POST /api/v1/decode
Decodifica un paquete cuántico.

**Body:**
```json
{
  "package_id": "BiMO-12345678",
  "measurements": [
    {"energia_medida": 5.2, "timestamp": "2025-01-01T10:00:00Z"},
    {"energia_medida": 3.8, "timestamp": "2025-01-01T10:00:01Z"}
  ]
}
```

**Respuesta exitosa (200):**
```json
{
  "status": "success",
  "data": {
    "package_id": "BiMO-12345678",
    "decoded_message": "MENSAJE DECODIFICADO",
    "transmission_status": "SUCCESS",
    "metrics": {
      "fidelidad": 0.95,
      "error_rate": 0.05,
      "coherencia": 0.92
    },
    "decoded_at": "2025-01-01T12:05:00Z"
  }
}
```

### GET /api/v1/packages
Obtiene el historial de paquetes del usuario.

**Respuesta:**
```json
{
  "status": "success",
  "data": {
    "packages": [
      {
        "package_id": "BiMO-12345678",
        "timestamp": "2025-01-01T12:00:00Z",
        "encoding_type": "BiMO",
        "qubits_count": 18,
        "status": "decoded",
        "metrics": {...}
      }
    ],
    "total_count": 1
  }
}
```

### GET /api/v1/packages/{package_id}
Obtiene detalles de un paquete específico.

## Códigos de Error

- `400` - Datos de entrada inválidos (`VALIDATION_ERROR`)
- `401` - No autorizado (`UNAUTHORIZED`, `INVALID_TOKEN`)
- `403` - Prohibido (`FORBIDDEN`)
- `404` - No encontrado (`PACKAGE_NOT_FOUND`, `NOT_FOUND`)
- `405` - Método no permitido (`METHOD_NOT_ALLOWED`)
- `500` - Error interno (`INTERNAL_ERROR`)

## Límites

- Mensaje máximo: 1000 caracteres
- Qubits máximos por mensaje: 100
- Timeout de simulación: 30 segundos

## Rate Limiting
En producción, se implementará rate limiting por usuario/IP.
"""

if __name__ == "__main__":
    print("=== Ejecutando ejemplos de QuantumLink ===")
    
    # Ejecutar ejemplo completo
    ejemplo_flujo_completo()
    
    # Ejecutar ejemplo de errores
    ejemplo_manejo_errores()
    
    print("\n=== Información adicional ===")
    print("- Revisa los archivos de configuración para personalizar la API")
    print("- Ejecuta 'pytest test_quantumlink_api.py -v' para tests automatizados")
    print("- Consulta la documentación de la API para más detalles")
    print("- Usa los ejemplos JavaScript para integración frontend")
    print("- Los comandos curl están listos para pruebas manuales"),
                'Content-Type': 'application/json'
            })
    
    def _make_request(self, method: str, endpoint: str, data: Dict = None) -> Dict[str, Any]:
        """Realizar una petición HTTP"""
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method.upper() == 'GET':
                response = self.session.get(url)
            elif method.upper() == 'POST':
                response = self.session.post(url, json=data)
            else:
                raise ValueError(f"Método HTTP no soportado: {method}")
            
            response_data = response.json()
            
            if response.status_code >= 400:
                raise Exception(f"Error API: {response_data.get('message', 'Error desconocido')}")
            
            return response_data
        
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error de conexión: {str(e)}")
    
    def health_check(self) -> Dict[str, Any]:
        """Verificar el estado de la API"""
        return self._make_request('GET', '/')
    
    def encode_message(self, message: str, encoding_type: str = "BiMO", priority: str = "medium") -> Dict[str, Any]:
        """Codificar un mensaje cuántico"""
        data = {
            "message": message,
            "options": {
                "encoding_type": encoding_type,
                "priority": priority
            }
        }
        return self._make_request('POST', '/api/v1/encode', data)
    
    def decode_message(self, package_id: str, measurements: List[Dict]) -> Dict[str, Any]:
        """Decodificar un paquete cuántico"""
        data = {
            "package_id": package_id,
            "measurements": measurements
        }
        return self._make_request('POST', '/api/v1/decode', data)
    
    def get_packages(self) -> Dict[str, Any]:
        """Obtener historial de paquetes"""
        return self._make_request('GET', '/api/v1/packages')
    
    def get_package_details(self, package_id: str) -> Dict[str, Any]:
        """Obtener detalles de un paquete específico"""
        return self._make_request('GET', f'/api/v1/packages/{package_id}')

# ==================== EJEMPLOS DE USO ====================

def ejemplo_flujo_completo():
    """Ejemplo completo del flujo de codificación y decodificación"""
    print("=== Ejemplo de Flujo Completo QuantumLink ===")
    
    # Inicializar cliente (token válido para pruebas)
    client = QuantumLinkClient(auth_token="valid_user123")
    
    try:
        # 1. Verificar estado de la API
        print("1. Verificando estado de la API...")
        health = client.health_check()
        print(f"   Estado: {health['message']}")
        
        # 2. Codificar un mensaje
        print("\n2. Codificando mensaje...")
        message = "QUANTUM COMMUNICATION IS THE FUTURE"
        encode_result = client.encode_message(
            message=message,
            encoding_type="BiMO",
            priority="high"
        )
        print(f"   Mensaje codificado exitosamente")
        print(f"   Package ID: {encode_result['data']['package_id']}")
        print(f"   Qubits utilizados: {encode_result['data']['qubits_count']}")
        
        package_id = encode_result['data']['package_id']
        
        # 3. Simular mediciones ruidosas (en un escenario real, vendrían del canal cuántico)
        print("\n3. Simulando mediciones cuánticas...")
        measurements = [
            {"energia_medida": 5.2, "timestamp": "2025-01-01T10:00:00Z"},
            {"energia_medida": 3.8, "timestamp": "2025-01-01T10:00:01Z"},
            {"energia_medida": 4.1, "timestamp": "2025-01-01T10:00:02Z"},
            {"energia_medida": 2.9, "timestamp": "2025-01-01T10:00:03Z"},
        ]
        
        # 4. Decodificar el mensaje
        print("4. Decodificando mensaje...")
        decode_result = client.decode_message(package_id, measurements)
        print(f"   Mensaje decodificado: {decode_result['data']['decoded_message']}")
        print(f"   Estado de transmisión: {decode_result['data']['transmission_status']}")
        print(f"   Métricas cuánticas:")
        for key, value in decode_result['data']['metrics'].items():
            print(f"     {key}: {value}")
        
        # 5. Obtener historial de paquetes
        print("\n5. Obteniendo historial...")
        packages = client.get_packages()
        print(f"   Total de paquetes: {packages['data']['total_count']}")
        
        # 6. Obtener detalles específicos del paquete
        print("6. Obteniendo detalles del paquete...")
        details = client.get_package_details(package_id)
        print(f"   Paquete encontrado: {details['data']['package']['id_mensaje']}")
        
    except Exception as e:
        print(f"Error: {str(e)}")

def ejemplo_manejo_errores():
    """Ejemplo de manejo de errores comunes"""
    print("\n=== Ejemplo de Manejo de Errores ===")
    
    client = QuantumLinkClient(auth_token="invalid_token")
    
    try:
        # Intentar codificar con token inválido
        client.encode_message("Test message")
    except Exception as e:
        print(f"Error esperado con token inválido: {str(e)}")
    
    # Cliente sin autenticación
    client_no_auth = QuantumLinkClient()
    try:
        client_no_auth.encode_message("Test message")
    except Exception as e:
        print(f"Error esperado sin autenticación: {str(e)}")

# ==================== EJEMPLOS JAVASCRIPT ====================

javascript_examples = """
// Ejemplo de cliente JavaScript para frontend

class QuantumLinkAPI {
    constructor(baseURL = 'http://localhost:5000', authToken = null) {
        this.baseURL = baseURL;
        this.authToken = authToken;
    }

    async makeRequest(method, endpoint, data = null) {
        const url = `${this.baseURL}${endpoint}`;
        const options = {
            method: method,
            headers: {
                'Content-Type': 'application/json',
            }
        };

        if (this.authToken) {
            options.headers['Authorization'] = `Bearer ${this.authToken}`;
        }

        if (data) {
            options.body = JSON.stringify(data);
        }

        try {
            const response = await fetch(url, options);
            const responseData = await response.json();

            if (!response.ok) {
                throw new Error(responseData.message || 'Error desconocido');
            }

            return responseData;
        } catch (error) {
            console.error('Error en la petición:', error);
            throw error;
        }
    }

    async encodeMessage(message, options = {}) {
        const data = {
            message: message,
            options: {
                encoding_type: options.encoding_type || 'BiMO',
                priority: options.priority || 'medium'
            }
        };
        return await this.makeRequest('POST', '/api/v1/encode', data);
    }

    async decodeMessage(packageId, measurements) {
        const data = {
            package_id: packageId,
            measurements: measurements
        };
        return await this.makeRequest('POST', '/api/v1/decode', data);
    }

    async getPackages() {
        return await this.makeRequest('GET', '/api/v1/packages');
    }
}

// Ejemplo de uso en React/Vue/Angular
async function ejemploFrontend() {
    const api = new QuantumLinkAPI('http://localhost:5000', 'valid_user123');

    try {
        // Codificar mensaje
        const encodeResult = await api.encodeMessage('HELLO QUANTUM WORLD');
        console.log('Mensaje codificado:', encodeResult);

        // Simular mediciones
        const measurements = [
            {energia_medida: 4.5, timestamp: new Date().toISOString()},
            {energia_medida: 3.2, timestamp: new Date().toISOString()}
        ];

        // Decodificar
        const decodeResult = await api.decodeMessage(
            encodeResult.data.package_id, 
            measurements
        );
        console.log('Mensaje decodificado:', decodeResult);

    } catch (error) {
        console.error('Error:', error.message);
    }
}
"""

# ==================== EJEMPLOS CURL ====================

curl_examples = """
# Ejemplos de comandos curl para probar la API

# 1. Verificar estado de la API
curl -X GET http://localhost:5000/

# 2. Codificar un mensaje
curl -X POST http://localhost:5000/api/v1/encode \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer valid_user123" \\
  -d '{
    "message": "QUANTUM TEST MESSAGE",
    "options": {
      "encoding_type": "BiMO",
      "priority": "high"
    }
  