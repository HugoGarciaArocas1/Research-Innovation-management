# Research & Innovation Management

Experimento de consumo energético de modelos de inteligencia artificial para la clasificación de neumonía en radiografías de tórax. El mismo script se ejecuta en distintos equipos para comparar precisión, tiempo de entrenamiento y emisiones.

Asignatura: **11751 — Gestió de la Investigació i la Innovació**, Grup 1 (Mallorca), Universitat de les Illes Balears.

Informe en Overleaf: [proyecto compartido](https://www.overleaf.com/project/6ac79bfe5785109c436213b4/share#1d6bca15f7a7bb1120aaa578a3b8b9b8061a457283114dfc)

## Qué hace el experimento

`Local_py_Hugo_dif_devices.py` entrena y evalúa dos clasificadores sobre el mismo conjunto de datos:

| Modelo | Descripción |
| --- | --- |
| **LightCNN** | Red convolucional ligera, definida en el propio script |
| **ResNet-18** | Arquitectura estándar, entrenada desde cero (`weights=None`) |

Durante el entrenamiento registra:

- precisión, precisión positiva, recall y F1 en el conjunto de test
- tiempo de entrenamiento, número de parámetros y FLOPs
- uso de CPU, RAM y, si hay GPU NVIDIA, utilización, memoria, temperatura y potencia
- emisiones estimadas con [CodeCarbon](https://codecarbon.io/)
- información del hardware del equipo (`hardware_info.json`)

El dispositivo de cálculo es CUDA si está disponible; en caso contrario, CPU. En macOS el script usa CPU.

## Requisitos

- Python 3.10 o superior
- Conexión a internet en la primera ejecución (descarga el dataset desde Hugging Face)
- Espacio en disco para las imágenes y los resultados (varios GB)
- GPU NVIDIA opcional. Sin ella el entrenamiento es más lento, sobre todo ResNet-18

## Preparar el entorno virtual

Abre una terminal en la carpeta del repositorio.

### macOS y Linux

```bash
cd Research-Innovation-management
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Windows (PowerShell)

```powershell
cd Research-Innovation-management
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Si PowerShell bloquea la activación, ejecuta una vez `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` y vuelve a activar el entorno.

El prompt debe mostrar `(.venv)`. Para salir del entorno:

```bash
deactivate
```

## Ejecutar el experimento

Con el entorno activado:

```bash
python Local_py_Hugo_dif_devices.py
```

La primera vez descarga `hf-vision/chest-xray-pneumonia`. Después entrena LightCNN y, a continuación, ResNet-18 (5 épocas, tamaño de imagen 224, batch 32, learning rate 0.001, semilla 42). Al terminar imprime una tabla resumen y la ruta de la carpeta `results/`.

## Resultados

Todos los archivos quedan en `results/`. Conserva la carpeta completa: sirve para comparar equipos.

| Archivo | Contenido |
| --- | --- |
| `experiment_summary.csv` | Tabla final: hardware, métricas, tiempo y emisiones |
| `hardware_info.json` | CPU, RAM, GPU, versiones de Python y PyTorch |
| `LightCNN_history.csv` / `ResNet18_history.csv` | Pérdida, accuracy y tiempo por época |
| `LightCNN_resources.csv` / `ResNet18_resources.csv` | Muestreo de CPU, RAM y GPU durante el entrenamiento |
| `LightCNN_confusion_matrix.csv` / `ResNet18_confusion_matrix.csv` | Matrices de confusión |
| `models/lightcnn.pth` / `models/resnet18.pth` | Pesos entrenados |
| `codecarbon/` | Registro de emisiones de CodeCarbon |

## Estructura del repositorio

```text
Research-Innovation-management/
├── Local_py_Hugo_dif_devices.py   # experimento reproducible entre equipos
├── requirements.txt
├── class_task_Main.ipynb
├── research&innovation_classTask1.ipynb
├── research&innovation_classTask1.1.ipynb
└── results/                       # se crea al ejecutar el script
```

Los notebooks recogen el trabajo de clase. El script de Python es la versión pensada para repetir el mismo experimento en cada ordenador.
