# 🏷️ Release v1.0.0 — hardsubextractor.exe

## Descripción

Presentamos la **primera versión** de `hardsubextractor.exe`, una herramienta desarrollada en Rust para la extracción automática de frames de texto en videos. Utiliza un modelo ONNX de segmentación y procesamiento eficiente con OpenCV, permitiendo identificar y guardar los segmentos donde aparece texto de manera precisa y rápida.

---

## Características Principales

- **Detección automática de texto** en videos mediante un modelo ONNX (segmentación tipo U-Net).
- **Procesamiento eficiente**: analiza cada N frames para optimizar recursos.
- **Extracción de segmentos estables**: guarda solo los frames representativos donde el texto es estable y relevante.
- **Configuración flexible**:
  - Umbral de cambio de texto ajustable.
  - Duración mínima para confirmar cambios de texto.
  - Opción de usar CPU o aceleradores (GPU, DirectML, ROCm, OpenVINO, CUDA).
- **Salida organizada**: los frames extraídos se guardan con nombres que indican el rango temporal del segmento detectado.
- **Soporte multiplataforma** (Windows, Linux).

---

## Funcionalidades

- Procesamiento de videos frame a frame para detectar la presencia de texto.
- Segmentación automática de períodos con texto estable.
- Guardado de frames representativos de cada segmento detectado.
- Limpieza automática del directorio de salida antes de cada ejecución.
- Métricas de tiempo y resumen al finalizar el procesamiento.

---

## Uso Básico

Ejecuta el binario directamente desde la terminal de Windows:

```powershell
./hardsubextractor.exe RUTA_DEL_VIDEO [MOSTRAR_FRAMES] [USAR_CPU]
```

- `RUTA_DEL_VIDEO`: Ruta al archivo de video a procesar (obligatorio).
- `MOSTRAR_FRAMES`: (opcional) `true`/`1` para mostrar frames (funcionalidad limitada).
- `USAR_CPU`: (opcional) `true`/`1` para forzar uso de CPU. Por defecto intenta usar aceleradores disponibles.

### Ejemplos

```powershell
# Procesar video usando GPU/aceleradores si están disponibles, sin mostrar frames
./hardsubextractor.exe videos/input.mp4

# Procesar video forzando CPU, sin mostrar frames
./hardsubextractor.exe videos/input.mp4 false true

# Procesar video, intentando mostrar frames (funcionalidad limitada) y usando GPU
./hardsubextractor.exe videos/input.mp4 true
```

---

## Notas

- Asegúrate de tener el modelo ONNX en `models/model.onnx`.
- Los frames extraídos se guardan en el directorio configurado (`extracted_text_frames_rust`).
- Consulta el README para detalles técnicos y configuración avanzada.

---

¡Gracias por probar la primera versión de `hardsubextractor.exe`!
