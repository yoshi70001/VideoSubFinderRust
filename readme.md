
# Extractor de Frames de Texto en Videos (Rust)

Este proyecto implementa una utilidad en Rust para extraer frames de videos que contienen texto. Utiliza un modelo ONNX para la detección de regiones de texto y OpenCV para el procesamiento de imágenes y video.

## Tabla de Contenidos

1.  [Visión General](#visión-general)
2.  [Dependencias Principales](#dependencias-principales)
3.  [Constantes de Configuración](#constantes-de-configuración)
4.  [Estructuras y Enums Clave](#estructuras-y-enums-clave)
    *   [`ExtractorError`](#extractonerror-enum)
    *   [`TextFrameExtractor`](#textframeextractor-struct)
5.  [Lógica Principal y Flujo de Trabajo](#lógica-principal-y-flujo-de-trabajo)
    *   [Inicialización](#inicialización)
    *   [Procesamiento de Video](#procesamiento-de-video)
    *   [Detección de Regiones y Máscara](#detección-de-regiones-y-máscara)
    *   [Manejo de Segmentos de Texto](#manejo-de-segmentos-de-texto)
6.  [Descripción del Modelo ONNX](#descripción-del-modelo-onnx)
7.  [Detalle de Métodos Importantes](#detalle-de-métodos-importantes)
    *   [`TextFrameExtractor::new()`](#textframeextracternew)
    *   [`TextFrameExtractor::process_video_file()`](#textframeextractorprocess_video_file)
    *   [`TextFrameExtractor::detect_regions_and_get_mask()`](#textframeextractordetect_regions_and_get_mask)
    *   [`TextFrameExtractor::preprocess_frame_for_onnx()`](#textframeextractorpreprocess_frame_for_onnx)
    *   [`TextFrameExtractor::postprocess_onnx_output()`](#textframeextractorpostprocess_onnx_output)
    *   [`TextFrameExtractor::calculate_mask_difference_percent()`](#textframeextractorcalculate_mask_difference_percent)
    *   [`TextFrameExtractor::save_segment_and_reset_monitoring()`](#textframeextractorsave_segment_and_reset_monitoring)
8.  [Función `main`](#función-main)
9.  [Uso](#uso)

## Visión General

El sistema procesa un archivo de video, frame por frame (o cada N frames), para identificar segmentos donde el texto está presente y es relativamente estable.
Cuando se detecta texto:
1.  Se guarda el primer frame del segmento.
2.  Si el texto desaparece o cambia significativamente (la máscara de texto cambia más allá de un umbral y esta nueva máscara persiste durante un tiempo mínimo), el segmento actual se cierra y se guarda el frame representativo.
3.  Si el cambio fue a una nueva configuración de texto, se inicia un nuevo segmento.

## Dependencias Principales

*   `opencv`: Para manipulación de imágenes y video (lectura, redimensionamiento, operaciones morfológicas, etc.).
*   `ort` (ONNX Runtime): Para ejecutar inferencias con el modelo ONNX de detección de texto.
*   `ndarray`: Para manipulación eficiente de arrays multidimensionales, necesarios para la entrada/salida del modelo ONNX.
*   `chrono`: Para formatear duraciones de tiempo para los nombres de archivo.
*   `thiserror`: Para la creación de tipos de error personalizados.

## Constantes de Configuración

*   `EXTRACTED_FRAMES_DIR_RUST`: `"extracted_text_frames_rust"` - Directorio donde se guardarán los frames extraídos.
*   `MASK_CHANGE_THRESHOLD_PERCENT_RUST`: `10.0` - Porcentaje de cambio en la máscara de texto necesario para considerar un cambio significativo.
*   `MIN_CHANGE_DURATION_MS_RUST`: `250` - Duración mínima (en milisegundos) que un cambio de máscara debe persistir para ser confirmado como un nuevo segmento.
*   `MODEL_PATH_RUST`: `"models/model.onnx"` - Ruta al archivo del modelo ONNX.

## Estructuras y Enums Clave

### `ExtractorError` (Enum)

Define los posibles errores que pueden ocurrir durante la ejecución.

```rust
#[derive(Debug, thiserror::Error)]
pub enum ExtractorError {
    #[error("OpenCV Error: {0}")]
    Cv(#[from] opencv::Error),
    #[error("ONNX Runtime Error: {0}")]
    Ort(#[from] OrtError),
    #[error("IO Error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Image Crate Error: {0}")] // Aunque image::ImageError se importa, no se usa activamente en el código provisto
    Image(#[from] image::ImageError),
    #[error("ndarray Shape Error: {0}")]
    NdarrayShape(#[from] ndarray::ShapeError),
    #[error("Path Error: {0}")]
    PathError(String),
    #[error("Video processing error: {0}")]
    VideoProcessing(String),
    #[error("Failed to get model input/output name")]
    ModelIONameError,
}
```

### `TextFrameExtractor` (Struct)

Contiene la configuración, la sesión ONNX y el estado necesario para procesar el video y extraer frames.

**Campos de Configuración e Inferencia:**
*   `output_dir: PathBuf`: Directorio de salida para los frames.
*   `mask_change_threshold: f64`: Umbral de cambio de máscara.
*   `min_change_duration: Duration`: Duración mínima para confirmar un cambio.
*   `session: Session`: Sesión de ONNX Runtime para la inferencia.
*   `input_name: String`: Nombre del tensor de entrada del modelo.
*   `output_name: String`: Nombre del tensor de salida del modelo.

**Campos de Estado (durante el procesamiento):**
*   `text_currently_present: bool`: Indica si se detecta texto en el frame actual.
*   `text_start_time: Option<Duration>`: Marca de tiempo de inicio del segmento de texto actual.
*   `last_known_text_end_time: Option<Duration>`: Última marca de tiempo donde se vio texto.
*   `frame_to_save_for_segment: Option<Mat>`: Frame representativo del segmento actual.
*   `mask_at_segment_start: Option<Mat>`: Máscara de texto al inicio del segmento actual.
*   `monitoring_mask_change: bool`: Indica si se está monitoreando un posible cambio de máscara.
*   `potential_new_mask_start_time: Option<Duration>`: Marca de tiempo de inicio de un posible nuevo segmento.
*   `potential_new_mask_candidate_frame: Option<Mat>`: Frame candidato para un nuevo segmento.
*   `potential_new_mask_candidate_mask: Option<Mat>`: Máscara candidata para un nuevo segmento.

## Lógica Principal y Flujo de Trabajo

### Inicialización

1.  Se crea el directorio de salida.
2.  Se configura la sesión de ONNX Runtime, eligiendo proveedores de ejecución (CPU o GPU/aceleradores).
3.  Se obtienen los nombres de los tensores de entrada y salida del modelo.

### Procesamiento de Video (`process_video_file`)

1.  Abre el archivo de video.
2.  Obtiene el FPS del video (o usa un valor por defecto si no está disponible).
3.  Itera sobre los frames del video (procesando cada 2º frame para optimizar).
    *   Calcula la marca de tiempo actual del frame.
    *   Llama a `detect_regions_and_get_mask()` para obtener la máscara de texto procesada y si se detectaron regiones.
    *   **Si hay regiones de texto:**
        *   Si el texto no estaba presente antes: Inicia un nuevo segmento de texto, guarda el frame actual y su máscara, y resetea el monitoreo de cambios.
        *   Si el texto ya estaba presente:
            *   Calcula la diferencia (`calculate_mask_difference_percent`) entre la máscara actual y la máscara al inicio del segmento.
            *   **Si la diferencia supera `mask_change_threshold`:**
                *   Si no se estaba monitoreando: Inicia el monitoreo, guardando el frame/máscara/tiempo actual como "potenciales" para un nuevo segmento.
                *   Si ya se estaba monitoreando: Verifica si la duración del cambio potencial (`current_time - potential_start_time`) supera `min_change_duration`.
                    *   Si se confirma el cambio: Guarda el segmento anterior (finalizándolo justo antes del `potential_start_time`), luego inicia un nuevo segmento con los datos "potenciales" guardados.
            *   **Si la diferencia NO supera el umbral:** Resetea el monitoreo si estaba activo (el cambio no persistió o fue insignificante).
    *   **Si NO hay regiones de texto:**
        *   Si el texto estaba presente antes: El texto acaba de desaparecer. Guarda el segmento actual (usando `previous_frame_time_with_text` como tiempo final) y resetea el estado.
        *   Resetea el monitoreo si estaba activo.
    *   Actualiza `last_known_text_end_time` si hay texto.
4.  Al finalizar el video, si un segmento de texto estaba activo, se guarda.

### Detección de Regiones y Máscara (`detect_regions_and_get_mask`)

1.  Preprocesa el frame BGR de entrada (`preprocess_frame_for_onnx`):
    *   Redimensiona a 224x224.
    *   Convierte de BGR a RGB.
    *   Normaliza los valores de píxeles a `[0, 1]`.
    *   Transpone de HWC (Alto, Ancho, Canales) a CHW.
    *   Añade una dimensión de batch (NCHW).
2.  Ejecuta la inferencia con la sesión ONNX.
3.  Postprocesa la salida del modelo ONNX (`postprocess_onnx_output`):
    *   Toma el tensor de salida (asume batch 1, 1 canal de salida).
    *   Crea una `Mat` OpenCV binaria (0 o 255) umbralizando la salida del modelo en 0.5.
4.  Aplica una operación de erosión a la máscara binaria para refinarla.
5.  Invierte la máscara (texto en blanco, fondo en negro -> texto en negro, fondo en blanco si `bitwise_not` está activado, o viceversa). *Nota: El código actual invierte la máscara antes de `find_contours`*.
6.  Encuentra contornos en la máscara erosionada (e invertida).
7.  Filtra los contornos por área para determinar si hay regiones de texto significativas (`has_regions`).
8.  Devuelve la máscara erosionada (antes de la inversión para contornos) y el booleano `has_regions`.

### Manejo de Segmentos de Texto

*   Un "segmento" es un período durante el cual el texto detectado es relativamente estable.
*   `save_segment_and_reset_monitoring()`:
    *   Formatea los tiempos de inicio y fin del segmento para el nombre del archivo.
    *   Guarda el `frame_to_save_for_segment` (el primer frame donde apareció este texto estable) en el `output_dir`.
    *   Resetea todas las variables de estado relacionadas con el segmento actual y el monitoreo de cambios.

## Descripción del Modelo ONNX

El modelo ONNX utilizado, según la arquitectura visualizada, es una red neuronal convolucional diseñada para tareas de segmentación semántica, probablemente una variante de U-Net.

**Estructura General:**

*   **Encoder (Parte Contractiva):**
    1.  `Input`: Se espera una imagen (preprocesada a NCHW, ej: 1x3x224x224).
    2.  Bloques Convolucionales: Múltiples capas `Conv` (con kernel 3x3) seguidas de activación `Relu`. Estos bloques extraen características de la imagen. La cantidad de filtros (canales de salida) típicamente aumenta (32 -> 64 -> 128 -> 256).
    3.  `MaxPool`: Capas de Max Pooling se intercalan para reducir la dimensionalidad espacial (downsampling) de los mapas de características y aumentar el campo receptivo.

*   **Bottleneck:**
    *   La parte más profunda del encoder (después del último MaxPool, con 256 filtros en el ejemplo) donde los mapas de características tienen la menor resolución espacial pero la mayor profundidad semántica.

*   **Decoder (Parte Expansiva):**
    1.  Bloques de Deconvolución/Transposición:
        *   `ConvTranspose`: Capas de convolución transpuesta (o deconvolución) para aumentar la dimensionalidad espacial (upsampling) de los mapas de características. En el diagrama, tienen un kernel 2x2.
    2.  `Concat`: Operaciones de concatenación que combinan los mapas de características upsampleados del decoder con los mapas de características correspondientes del encoder (conexiones skip o residuales). Esto es característico de las arquitecturas U-Net y ayuda a preservar detalles de baja resolución y mejorar la localización.
    3.  Bloques Convolucionales: Similar al encoder, capas `Conv` + `Relu` para refinar los mapas de características combinados. La cantidad de filtros típicamente disminuye (256 -> 128 -> 64 -> 32).

*   **Output Layer:**
    1.  Una capa `Conv` final, típicamente con un kernel 1x1, para mapear los mapas de características del último bloque del decoder al número deseado de clases de salida. En este caso, para una máscara de segmentación binaria, la salida es un solo canal (W: `<1x32x1x1>`).
    2.  `output`: El resultado es un mapa de segmentación (ej: 1x1x224x224), donde cada píxel tiene un valor que indica la probabilidad de pertenecer a la clase "texto".

**Flujo de Datos:**
La imagen de entrada pasa por el encoder, disminuyendo su tamaño espacial y aumentando su profundidad. Luego, el decoder reconstruye la resolución espacial, utilizando la información de las conexiones skip del encoder para producir una máscara de segmentación precisa del mismo tamaño que la entrada (o un tamaño relacionado).

**Propósito:**
El modelo está diseñado para identificar y segmentar regiones de texto en una imagen de entrada, produciendo una máscara binaria donde los píxeles de texto están marcados.

```
[Input]
   |
   V
Conv (32 filtros, 3x3) -> Relu
   |
   V
Conv (32 filtros, 3x3) -> Relu  ---------------------------------------> Concat
   |                                                                       ^
   V                                                                       |
MaxPool                                                                    |
   |                                                                       |
   V                                                                       |
Conv (64 filtros, 3x3) -> Relu                                             |
   |                                                                       |
   V                                                                       |
Conv (64 filtros, 3x3) -> Relu  ---------------------------------> Concat  |
   |                                                                  ^    |
   V                                                                  |    |
MaxPool                                                               |    |
   |                                                                  |    |
   V                                                                  |    |
Conv (128 filtros, 3x3) -> Relu                                        |    |
   |                                                                  |    |
   V                                                                  |    |
Conv (128 filtros, 3x3) -> Relu ---------------------------> Concat   |    |
   |                                                             ^     |    |
   V                                                             |     |    |
MaxPool                                                          |     |    |
   |                                                             |     |    |
   V                                                             |     |    |
Conv (256 filtros, 3x3) -> Relu                                   |     |    |
   |                                                             |     |    |
   V                                                             |     |    |
Conv (256 filtros, 3x3) -> Relu                                   |     |    |
   |                                                             |     |    |
   V                                                             |     |    |
ConvTranspose (128 filtros, 2x2) --------------------------------      |    |
   |                                                                   |    |
   V                                                                   |    |
(Después de Concat) Conv (128 filtros, 3x3) -> Relu                     |    |
   |                                                                   |    |
   V                                                                   |    |
Conv (128 filtros, 3x3) -> Relu                                         |    |
   |                                                                   |    |
   V                                                                   |    |
ConvTranspose (64 filtros, 2x2) ---------------------------------------     |
   |                                                                        |
   V                                                                        |
(Después de Concat) Conv (64 filtros, 3x3) -> Relu                          |
   |                                                                        |
   V                                                                        |
Conv (64 filtros, 3x3) -> Relu                                              |
   |                                                                        |
   V                                                                        |
ConvTranspose (32 filtros, 2x2) --------------------------------------------
   |
   V
(Después de Concat) Conv (32 filtros, 3x3) -> Relu
   |
   V
Conv (32 filtros, 3x3) -> Relu
   |
   V
Conv (1 filtro, 1x1)
   |
   V
[Output (Máscara de segmentación)]
```

## Detalle de Métodos Importantes

### `TextFrameExtractor::new()`

```rust
pub fn new(
    model_path: &Path,
    output_dir: &Path,
    mask_change_threshold: f64,
    min_change_duration_ms: u64,
    use_cpu: bool,
) -> Result<Self>
```
Constructor. Inicializa la sesión ONNX, establece los directorios y umbrales.
*   `model_path`: Ruta al archivo `.onnx`.
*   `output_dir`: Dónde guardar los frames.
*   `mask_change_threshold`: Umbral de diferencia porcentual de la máscara.
*   `min_change_duration_ms`: Tiempo mínimo que un cambio de máscara debe persistir.
*   `use_cpu`: Si es `true`, usa solo CPU; de lo contrario, intenta usar aceleradores (ROCm, CUDA, CoreML, etc.).

### `TextFrameExtractor::process_video_file()`

```rust
pub fn process_video_file(&mut self, video_path: &Path, _display_frames: bool) -> Result<()>
```
Método principal que orquesta la lectura del video, la detección de texto por frame y el guardado de segmentos.
*   `video_path`: Ruta al archivo de video.
*   `_display_frames`: Booleano para mostrar frames (actualmente no implementado completamente en el bucle).

### `TextFrameExtractor::detect_regions_and_get_mask()`

```rust
fn detect_regions_and_get_mask(&self, frame_bgr: &Mat) -> Result<(Mat, bool)>
```
Realiza la inferencia del modelo en un frame y determina si contiene regiones de texto.
1.  Preprocesa el frame.
2.  Ejecuta la sesión ONNX.
3.  Postprocesa la salida para obtener una máscara binaria.
4.  Aplica erosión a la máscara.
5.  Busca contornos y los filtra por área.
Devuelve la máscara erosionada y un booleano indicando si se encontraron regiones válidas.

### `TextFrameExtractor::preprocess_frame_for_onnx()`

```rust
fn preprocess_frame_for_onnx(&self, frame_bgr: &Mat) -> Result<Array<f32, Ix4>>
```
Prepara un frame de OpenCV (`Mat` en formato BGR) para la entrada del modelo ONNX.
1.  Redimensiona a 224x224 (INTER_LINEAR).
2.  Convierte de BGR a RGB.
3.  Convierte los datos de `u8` a `f32` y normaliza dividiendo por 255.0.
4.  Reorganiza los datos de HWC (Alto, Ancho, Canales) a CHW (Canales, Alto, Ancho).
5.  Añade una dimensión de batch (NCHW), resultando en `Ix4` (1, Canales, Alto, Ancho).

### `TextFrameExtractor::postprocess_onnx_output()`

```rust
fn postprocess_onnx_output(
    &self,
    onnx_output_single_frame: ndarray::ArrayView3<f32>, // CHW (1xHxW)
) -> Result<Mat>
```
Convierte la salida cruda del modelo ONNX (un tensor `f32`) en una máscara binaria de OpenCV (`Mat` `CV_8UC1`).
1.  Asume que la salida es (1, Alto, Ancho) después de quitar la dimensión de batch.
2.  Toma el único canal.
3.  Crea una `Mat` de OpenCV de tipo `CV_8UC1` (un solo canal, 8 bits sin signo).
4.  Itera sobre los valores del tensor de salida: si el valor es `> 0.5`, el píxel correspondiente en la `Mat` se establece en `255`, de lo contrario en `0`.

### `TextFrameExtractor::calculate_mask_difference_percent()`

```rust
fn calculate_mask_difference_percent(
    &self,
    mask1_opt: Option<&Mat>,
    mask2: &Mat,
) -> Result<f64>
```
Calcula el porcentaje de píxeles diferentes entre dos máscaras.
1.  Si `mask1_opt` es `None` (no hay máscara previa), devuelve 0.0% o 100.0% (actualmente 0.0%).
2.  Si las máscaras están vacías o tienen diferentes tamaños/tipos, devuelve 0.0% o 100.0%.
3.  Calcula la diferencia absoluta (`core::absdiff`).
4.  Cuenta los píxeles no cero en la diferencia (`core::count_non_zero`).
5.  Devuelve `(non_zero_diff_pixels / total_pixels) * 100.0`.

### `TextFrameExtractor::save_segment_and_reset_monitoring()`

```rust
fn save_segment_and_reset_monitoring(&mut self, end_time_for_saving: Duration) -> Result<()>
```
Guarda el frame representativo del segmento actual y resetea el estado para el siguiente.
1.  Formatea los tiempos de inicio y fin para el nombre del archivo: `HH_MM_SS_sss___HH_MM_SS_sss.jpeg`.
2.  Guarda el `frame_to_save_for_segment` usando `imgcodecs::imwrite`.
3.  Resetea todas las variables de estado `text_start_time`, `frame_to_save_for_segment`, `mask_at_segment_start`, y las variables de monitoreo de cambio.

## Función `main`

```rust
fn main() -> Result<()>
```
El punto de entrada del programa.
1.  Parsea argumentos de línea de comandos:
    *   `path`: Ruta al video.
    *   `display_frames` (opcional, bool): Si se deben mostrar los frames.
    *   `use_cpu` (opcional, bool): Si se debe forzar el uso de CPU.
2.  Verifica la existencia del archivo de video y del modelo.
3.  Crea una instancia de `TextFrameExtractor`.
4.  Limpia el directorio de salida (`clear_output_directory`) y cualquier archivo `.srt` antiguo.
5.  Llama a `extractor.process_video_file()` para iniciar el procesamiento.
6.  Imprime métricas de tiempo y la ubicación de los frames extraídos.

## Uso

Para compilar y ejecutar:

1.  **Compilar:**
    ```bash
    cargo build --release
    ```
2.  **Ejecutar:**
    El ejecutable estará en `target/release/nombre_del_binario`.
    ```bash
    ./target/release/nombre_del_binario RUTA_DEL_VIDEO [MOSTRAR_FRAMES] [USAR_CPU]
    ```
    *   `RUTA_DEL_VIDEO`: Obligatorio. Ejemplo: `videos/mi_video.mp4`.
    *   `MOSTRAR_FRAMES`: Opcional. `true` o `1` para activar (actualmente con funcionalidad limitada en el bucle principal). Por defecto `false`.
    *   `USAR_CPU`: Opcional. `true` o `1` para forzar CPU. Por defecto `false` (intenta usar GPU/aceleradores).

    **Ejemplos:**
    ```bash
    # Procesar video usando GPU/aceleradores si están disponibles, sin mostrar frames
    ./target/release/tu_programa_rust videos/input.mp4

    # Procesar video forzando CPU, sin mostrar frames
    ./target/release/tu_programa_rust videos/input.mp4 false true

    # Procesar video, intentando mostrar frames (funcionalidad limitada) y usando GPU
    ./target/release/tu_programa_rust videos/input.mp4 true
    ```

**Nota:** Asegúrate de que el modelo ONNX (`models/model.onnx`) esté en la ubicación correcta o ajusta `MODEL_PATH_RUST`.
