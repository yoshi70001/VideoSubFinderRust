use chrono::Duration as ChronoDuration;
use ndarray::{Array, Axis, Ix4, s};
use opencv::{
    core::{self, CV_8UC1, Mat, Size},
    highgui,
    imgcodecs,
    imgproc,
    prelude::*, // For MatTrait, MatTraitConst
    videoio,
};
use ort::Error as OrtError;
use ort::execution_providers::{
    CPUExecutionProvider, CUDAExecutionProvider, CoreMLExecutionProvider,
    DirectMLExecutionProvider, OpenVINOExecutionProvider, ROCmExecutionProvider,
};
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use std::fs;
use std::path::{Path, PathBuf};
use std::thread::available_parallelism;
use std::time::Duration; // For formatting
const EXTRACTED_FRAMES_DIR_RUST: &str = "extracted_text_frames_rust";
const MASK_CHANGE_THRESHOLD_PERCENT_RUST: f64 = 10.0;
const MIN_CHANGE_DURATION_MS_RUST: u64 = 250;
const MODEL_PATH_RUST: &str = "models/model.onnx"; // Ensure this model exists

#[derive(Debug, thiserror::Error)]
pub enum ExtractorError {
    #[error("OpenCV Error: {0}")]
    Cv(#[from] opencv::Error),
    #[error("ONNX Runtime Error: {0}")]
    Ort(#[from] OrtError),
    #[error("IO Error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Image Crate Error: {0}")]
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

type Result<T> = std::result::Result<T, ExtractorError>;

pub struct TextFrameExtractor {
    output_dir: PathBuf,
    mask_change_threshold: f64,
    min_change_duration: Duration,
    session: Session,
    input_name: String,
    output_name: String,

    // State variables
    text_currently_present: bool,
    text_start_time: Option<Duration>,
    last_known_text_end_time: Option<Duration>,
    frame_to_save_for_segment: Option<Mat>,
    mask_at_segment_start: Option<Mat>,

    monitoring_mask_change: bool,
    potential_new_mask_start_time: Option<Duration>,
    potential_new_mask_candidate_frame: Option<Mat>,
    potential_new_mask_candidate_mask: Option<Mat>,
}

impl TextFrameExtractor {
    pub fn new(
        model_path: &Path,
        output_dir: &Path,
        mask_change_threshold: f64,
        min_change_duration_ms: u64,
        use_cpu: bool,
    ) -> Result<Self> {
        fs::create_dir_all(output_dir)?;

        let providers = if use_cpu {
            vec![CPUExecutionProvider::default().build()]
        } else {
            vec![
                ROCmExecutionProvider::default().build(),
                CUDAExecutionProvider::default().build(),
                CoreMLExecutionProvider::default().build(),
                OpenVINOExecutionProvider::default().build(),
                DirectMLExecutionProvider::default().build(),
            ]
        };

        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Disable)?
            .with_parallel_execution(true)? // If desired and ort version supports
            .with_intra_threads(available_parallelism()?.get())?
            .with_execution_providers(providers)?
            .commit_from_file(model_path)?;

        let input_name = session.inputs[0].name.clone();
        let output_name = session.outputs[0].name.clone();

        if input_name.is_empty() || output_name.is_empty() {
            return Err(ExtractorError::ModelIONameError);
        }

        Ok(Self {
            output_dir: output_dir.to_path_buf(),
            mask_change_threshold,
            min_change_duration: Duration::from_millis(min_change_duration_ms),
            session,
            input_name,
            output_name,
            text_currently_present: false,
            text_start_time: None,
            last_known_text_end_time: None,
            frame_to_save_for_segment: None,
            mask_at_segment_start: None,
            monitoring_mask_change: false,
            potential_new_mask_start_time: None,
            potential_new_mask_candidate_frame: None,
            potential_new_mask_candidate_mask: None,
        })
    }

    fn format_duration(duration_opt: Option<Duration>) -> String {
        if let Some(d) = duration_opt {
            let cd = ChronoDuration::from_std(d).unwrap_or_else(|_| ChronoDuration::zero());
            let hours = cd.num_hours();
            let minutes = cd.num_minutes() % 60;
            let seconds = cd.num_seconds() % 60;
            let milliseconds = cd.num_milliseconds() % 1000;
            format!(
                "{:02}_{:02}_{:02}_{:03}",
                hours, minutes, seconds, milliseconds
            )
        } else {
            "00_00_00_000".to_string()
        }
    }

    fn preprocess_frame_for_onnx(&self, frame_bgr: &Mat) -> Result<Array<f32, Ix4>> {
        let mut resized_frame = Mat::default();
        imgproc::resize(
            frame_bgr,
            &mut resized_frame,
            Size::new(224, 224),
            0.0,
            0.0,
            imgproc::INTER_LINEAR,
        )?;

        let mut rgb_frame = Mat::default();
        imgproc::cvt_color(&resized_frame, &mut rgb_frame, imgproc::COLOR_BGR2RGB, 0)?;

        let rows = rgb_frame.rows() as usize;
        let cols = rgb_frame.cols() as usize;
        let channels = rgb_frame.channels() as usize; // Debería ser 3

        if channels != 3 {
            return Err(ExtractorError::VideoProcessing(
                "Expected 3 channels for RGB after cvt_color".into(),
            ));
        }

        let array_hwc: Array<f32, _> = if rgb_frame.is_continuous() {
            // Opción más rápida: Mat es continua, podemos obtener un slice de todos los datos
            let data_slice_u8 = unsafe {
                // rgb_frame.data() devuelve *const u8
                std::slice::from_raw_parts(rgb_frame.data(), rows * cols * channels)
            };

            // Convertir el slice de u8 a un iterador de f32 y normalizar
            // Asumiendo que data_slice_u8 está en orden R,G,B,R,G,B... porque hicimos cvt_color a rgb_frame
            let float_iter = data_slice_u8.iter().map(|&val_u8| (val_u8 as f32) / 255.0);

            // Crear el ndarray desde el iterador
            // Esto crea un array 1D, luego lo remoldeamos.
            Array::from_iter(float_iter).into_shape((rows, cols, channels))? // H, W, C
        } else {
            // Opción más lenta (fallback): Mat no es continua, iterar por filas
            // Todavía es mejor que at_2d por píxel.
            let mut flat_data = Vec::with_capacity(rows * cols * channels);
            for r in 0..rows {
                // rgb_frame.ptr(r as i32)? devuelve *const u8 para esa fila
                let row_slice_u8 = unsafe {
                    std::slice::from_raw_parts(rgb_frame.ptr(r as i32)?, cols * channels)
                };
                for i in (0..row_slice_u8.len()).step_by(channels) {
                    // El orden es R, G, B porque rgb_frame es el resultado de COLOR_BGR2RGB
                    flat_data.push((row_slice_u8[i] as f32) / 255.0); // R
                    flat_data.push((row_slice_u8[i + 1] as f32) / 255.0); // G
                    flat_data.push((row_slice_u8[i + 2] as f32) / 255.0); // B
                }
            }
            Array::from_shape_vec((rows, cols, channels), flat_data)?
        };

        // Transpose HWC to CHW
        let array_chw = array_hwc.permuted_axes([2, 0, 1]).to_owned();

        // Add batch dimension: CHW -> NCHW
        Ok(array_chw.insert_axis(Axis(0)))
    }

    fn postprocess_onnx_output(
        &self,
        onnx_output_single_frame: ndarray::ArrayView3<f32>, // CHW (1xHxW)
    ) -> Result<Mat> {
        let (channels_out, height_out, width_out) = onnx_output_single_frame.dim();

        if channels_out != 1 {
            return Err(ExtractorError::VideoProcessing(format!(
                "Expected 1 output channel from model, got {}",
                channels_out
            )));
        }

        // output_single_frame es 1xHxW. Tomamos el primer (y único) canal.
        let single_channel_view = onnx_output_single_frame.slice(s![0, .., ..]); // HxW

        // Crear la Mat de salida
        let mut output_mat = Mat::new_rows_cols_with_default(
            height_out as i32,
            width_out as i32,
            CV_8UC1, // Single channel u8
            core::Scalar::all(0.0),
        )?;

        // Asegurarse de que output_mat sea continua para el acceso rápido si es posible.
        // (new_rows_cols_with_default usualmente lo es, pero una verificación no hace daño
        // o forzarla con .clone_continuous() si es necesario, aunque aquí podría no ser tan crítico)

        if output_mat.is_continuous() && single_channel_view.is_standard_layout() {
            // Acceso rápido si ambos son continuos/estándar
            let mat_data_slice_u8_mut = unsafe {
                // output_mat.data_mut() devuelve *mut u8
                std::slice::from_raw_parts_mut(output_mat.data_mut(), height_out * width_out)
            };

            // single_channel_view (HxW) ya está en el orden correcto para iterar
            // y llenar mat_data_slice_u8_mut secuencialmente.
            for (f_val, mat_pixel_ref) in single_channel_view
                .iter()
                .zip(mat_data_slice_u8_mut.iter_mut())
            {
                *mat_pixel_ref = if *f_val > 0.5 { 255 } else { 0 };
            }
        } else {
            // Fallback: iterar con at_2d_mut (como en tu código original)
            // O un bucle por filas si solo la Mat no es continua.
            for r in 0..height_out {
                for c in 0..width_out {
                    let val = single_channel_view[[r, c]];
                    *output_mat.at_2d_mut::<u8>(r as i32, c as i32)? =
                        if val > 0.5 { 255 } else { 0 };
                }
            }
        }
        Ok(output_mat)
    }
    fn calculate_mask_difference_percent(
        &self,
        mask1_opt: Option<&Mat>,
        mask2: &Mat,
    ) -> Result<f64> {
        let mask1 = match mask1_opt {
            Some(m) => m,
            None => return Ok(0.0), // Or 100.0 if no prior mask means full difference
        };

        if mask1.empty() || mask2.empty() {
            return Ok(0.0);
        }
        if mask1.size()? != mask2.size()? || mask1.typ() != mask2.typ() {
            return Ok(100.0); // Different shapes/types means 100% difference
        }

        let mut diff = Mat::default();
        core::absdiff(mask1, mask2, &mut diff)?;

        let non_zero_diff_pixels = core::count_non_zero(&diff)?;
        let total_pixels = mask1.total();

        if total_pixels == 0 {
            Ok(0.0)
        } else {
            Ok((non_zero_diff_pixels as f64 / total_pixels as f64) * 100.0)
        }
    }

    fn detect_regions_and_get_mask(&self, frame_bgr: &Mat) -> Result<(Mat, bool)> {
        let input_data = self.preprocess_frame_for_onnx(frame_bgr)?;

        let outputs_ort = self
            .session
            .run(ort::inputs![self.input_name.as_str() => input_data.view()]?)?;
        let onnx_output_tensor =
            outputs_ort[self.output_name.as_str()].try_extract_tensor::<f32>()?;

        // Assuming batch size is 1, so we take the first element.
        let output_mask_raw_onnx = onnx_output_tensor.slice(s![0, .., .., ..]); // CHW view

        let output_mask_processed = self.postprocess_onnx_output(output_mask_raw_onnx)?;

        let kernel_erode = imgproc::get_structuring_element(
            imgproc::MORPH_RECT,
            Size::new(6, 3), // Python had (3,6), OpenCV is (cols, rows) for Size
            core::Point::new(-1, -1),
        )?;
        let mut infered_frame_eroded = Mat::default();
        imgproc::erode(
            &output_mask_processed,
            &mut infered_frame_eroded,
            &kernel_erode,
            core::Point::new(-1, -1),
            2, // iterations
            core::BORDER_CONSTANT,
            imgproc::morphology_default_border_value()?,
        )?;

        // In Python: processed_mask_for_contours = cv2.bitwise_not(infered_frame_eroded)
        // This implies contours are found on white objects on black background.
        // If infered_frame_eroded has text as white, then no bitwise_not is needed.
        // If text is black, then bitwise_not. Let's assume text is white (255).
        // For findContours, the input image is modified, so clone if needed later.
        // Let's assume the mask `infered_frame_eroded` has text regions as WHITE (255).
        let mut contours_mask = infered_frame_eroded.clone(); // findContours modifies input
        // If you want to invert the mask, uncomment the next line:
        core::bitwise_not(&infered_frame_eroded, &mut contours_mask, &Mat::default())?;
        // highgui::imshow("contours_mask", &contours_mask)?;
        let mut contours = opencv::core::Vector::<opencv::core::Vector<core::Point>>::new();
        imgproc::find_contours(
            &mut contours_mask, // Input image, will be modified
            &mut contours,
            imgproc::RETR_EXTERNAL,
            imgproc::CHAIN_APPROX_SIMPLE,
            core::Point::new(0, 0),
        )?;

        let mut has_regions = false;
        if !contours.is_empty() {
            for contour in contours.iter() {
                let area = imgproc::contour_area(&contour, false)?;
                // Python: 240 < area < (224 * 112 * 0.9)
                // 224*112*0.9 = 22579.2
                if area > 240.0 && area < 22580.0 {
                    // Adjust as needed
                    has_regions = true;
                    break;
                }
            }
        }
        // The mask returned should be the one used for diff calculation later,
        // so `infered_frame_eroded` seems appropriate.
        Ok((infered_frame_eroded, has_regions))
    }

    fn save_segment_and_reset_monitoring(&mut self, end_time_for_saving: Duration) -> Result<()> {
        if let (Some(frame_to_save), Some(start_time)) =
            (&self.frame_to_save_for_segment, self.text_start_time)
        {
            let mut actual_end_time = end_time_for_saving;
            if start_time > end_time_for_saving {
                eprintln!(
                    "Advertencia: Tiempo de inicio del segmento ({:?}) es posterior al tiempo final calculado ({:?}).",
                    start_time, end_time_for_saving
                );
                // Fallback: make segment last at least ~40ms
                actual_end_time = start_time + Duration::from_millis(40);
                eprintln!("           Ajustando tiempo final a: {:?}", actual_end_time);
            }

            let start_str = Self::format_duration(Some(start_time));
            let end_str = Self::format_duration(Some(actual_end_time));

            let filename = format!("{}___{}.jpeg", start_str, end_str);
            let filepath = self.output_dir.join(filename);

            imgcodecs::imwrite(
                filepath.to_str().ok_or_else(|| {
                    ExtractorError::PathError("Invalid filepath for saving".into())
                })?,
                frame_to_save,
                &opencv::core::Vector::<i32>::new(),
            )?;
            // println!("Guardado localmente: {:?}", filepath);
        }

        // Reset segment state
        self.text_start_time = None;
        self.frame_to_save_for_segment = None;
        self.mask_at_segment_start = None;

        // Reset monitoring state
        self.monitoring_mask_change = false;
        self.potential_new_mask_start_time = None;
        self.potential_new_mask_candidate_frame = None;
        self.potential_new_mask_candidate_mask = None;

        Ok(())
    }

    pub fn process_video_file(&mut self, video_path: &Path, _display_frames: bool) -> Result<()> {
        let mut cap = videoio::VideoCapture::from_file(
            video_path
                .to_str()
                .ok_or_else(|| ExtractorError::PathError("Invalid video path".into()))?,
            videoio::CAP_ANY,
        )?;
        if !cap.is_opened()? {
            return Err(ExtractorError::VideoProcessing(format!(
                "Cannot open video: {:?}",
                video_path
            )));
        }

        let fps = cap.get(videoio::CAP_PROP_FPS)?;
        let fps = if fps > 0.0 {
            fps
        } else {
            eprintln!(
                "Advertencia: FPS del video es {}. Usando 25 FPS por defecto.",
                fps
            );
            25.0
        };

        let mut frame_count: u64 = 0;
        let mut current_frame_mat = Mat::default();
        let mut previous_frame_time_with_text: Duration = Duration::from_secs(0);

        loop {
            if !cap.read(&mut current_frame_mat)? || current_frame_mat.empty() {
                break;
            }

            if frame_count % 2 == 0 {
                // Process every 2nd frame like in Python
                let current_time = Duration::from_secs_f64(frame_count as f64 / fps);
                let (current_processed_mask, frame_has_region) =
                    self.detect_regions_and_get_mask(&current_frame_mat)?;

                if frame_has_region {
                    if !self.text_currently_present {
                        // Text appears
                        self.text_currently_present = true;
                        self.text_start_time = Some(current_time);
                        self.mask_at_segment_start = Some(current_processed_mask.clone());
                        self.frame_to_save_for_segment = Some(current_frame_mat.clone());

                        self.monitoring_mask_change = false;
                        self.potential_new_mask_start_time = None;
                        self.potential_new_mask_candidate_frame = None;
                        self.potential_new_mask_candidate_mask = None;
                    } else {
                        // Text continues, check for mask change
                        let diff_percent = self.calculate_mask_difference_percent(
                            self.mask_at_segment_start.as_ref(),
                            &current_processed_mask,
                        )?;

                        if diff_percent > self.mask_change_threshold {
                            if !self.monitoring_mask_change {
                                // Significant change, start monitoring
                                self.monitoring_mask_change = true;
                                self.potential_new_mask_start_time = Some(current_time);
                                self.potential_new_mask_candidate_frame =
                                    Some(current_frame_mat.clone());
                                self.potential_new_mask_candidate_mask =
                                    Some(current_processed_mask.clone());
                            } else {
                                // Already monitoring, check if persisted
                                if let Some(potential_start) = self.potential_new_mask_start_time {
                                    if (current_time - potential_start) >= self.min_change_duration
                                    {
                                        // Change confirmed
                                        let confirmed_start_time_for_new_segment =
                                            self.potential_new_mask_start_time.unwrap(); // Safe due to above
                                        let confirmed_mask_for_new_segment =
                                            self.potential_new_mask_candidate_mask.clone().unwrap();
                                        let confirmed_frame_for_new_segment = self
                                            .potential_new_mask_candidate_frame
                                            .clone()
                                            .unwrap();

                                        let time_offset_for_prev_segment =
                                            Duration::from_secs_f64(1.0 / fps)
                                                .max(Duration::from_millis(1));
                                        let end_time_of_previous_segment =
                                            confirmed_start_time_for_new_segment
                                                .checked_sub(time_offset_for_prev_segment)
                                                .unwrap_or_else(|| Duration::from_secs(0)); // Prevent underflow

                                        self.save_segment_and_reset_monitoring(
                                            end_time_of_previous_segment,
                                        )?;

                                        // Start new segment
                                        self.text_currently_present = true; // Should already be true
                                        self.text_start_time =
                                            Some(confirmed_start_time_for_new_segment);
                                        self.mask_at_segment_start =
                                            Some(confirmed_mask_for_new_segment);
                                        self.frame_to_save_for_segment =
                                            Some(confirmed_frame_for_new_segment);
                                        // Monitoring vars were reset by save_segment_and_reset_monitoring
                                    }
                                    // Else: monitoring, but not persisted long enough
                                }
                            }
                        } else {
                            // No significant change
                            if self.monitoring_mask_change {
                                // Was monitoring, but change reverted or was insignificant
                                self.monitoring_mask_change = false;
                                self.potential_new_mask_start_time = None;
                                self.potential_new_mask_candidate_frame = None;
                                self.potential_new_mask_candidate_mask = None;
                            }
                        }
                    }
                    previous_frame_time_with_text = current_time;
                } else {
                    // No region in current frame
                    if self.text_currently_present {
                        // Text just disappeared
                        self.save_segment_and_reset_monitoring(previous_frame_time_with_text)?;
                        self.text_currently_present = false;
                    }
                    if self.monitoring_mask_change {
                        self.monitoring_mask_change = false;
                        self.potential_new_mask_start_time = None;
                        self.potential_new_mask_candidate_frame = None;
                        self.potential_new_mask_candidate_mask = None;
                    }
                }
                if self.text_currently_present {
                    self.last_known_text_end_time = Some(current_time);
                }

                // TODO: Implement display_frames if needed, using opencv::highgui
                if _display_frames {
                    highgui::imshow("Frame", &current_frame_mat)?;
                    highgui::wait_key(1)?;
                }
            }
            // println!(
            //     "Procesando frame {} a tiempo: {:.2} segundos",
            //     frame_count,
            //     frame_count as f64 / fps
            // );
            frame_count += 1;
        }

        if self.text_currently_present {
            if let Some(last_time) = self.last_known_text_end_time {
                self.save_segment_and_reset_monitoring(last_time)?;
            }
        }

        cap.release()?;
        // if display_frames { highgui::destroy_all_windows()?; }
        println!("Procesamiento de video y extracción de frames completado (Rust).");
        Ok(())
    }

    pub fn clear_output_directory(&self) -> Result<()> {
        if self.output_dir.exists() {
            println!("Limpiando directorio: {:?}", self.output_dir);
            fs::remove_dir_all(&self.output_dir)?;
        }
        fs::create_dir_all(&self.output_dir)?;
        Ok(())
    }
}

fn main() -> Result<()> {
    struct Cli {
        use_cpu: bool,
        display_frames: bool,
        path: std::path::PathBuf,
    }
    let use_cpu = std::env::args()
        .nth(3)
        .map_or(false, |arg| arg.to_lowercase() == "true" || arg == "1");
    let display_frames = std::env::args()
        .nth(2)
        .map_or(false, |arg| arg.to_lowercase() == "true" || arg == "1");
    let path = std::env::args()
        .nth(1)
        .expect("No definio el path del video");

    let args = Cli {
        use_cpu: use_cpu,
        display_frames: display_frames,
        path: std::path::PathBuf::from(path),
    };

    println!(
        "display_frames: {:?}, path: {:?}, use_cpu: {:?}",
        args.display_frames, args.path, args.use_cpu
    );
    let video_file_path_str = args
        .path
        .to_str()
        .ok_or_else(|| ExtractorError::PathError("Invalid video file path".into()))?;
    let video_file_path = Path::new(video_file_path_str);

    if !video_file_path.exists() {
        eprintln!(
            "Error: El archivo de video '{}' no fue encontrado.",
            video_file_path_str
        );
        return Ok(()); // Or return an error
    }

    let model_path = Path::new(MODEL_PATH_RUST);
    if !model_path.exists() {
        eprintln!(
            "Error: El archivo de modelo ONNX '{:?}' no fue encontrado.",
            model_path
        );
        return Ok(());
    }

    let output_dir = Path::new(EXTRACTED_FRAMES_DIR_RUST);

    let start_time_total = std::time::Instant::now();

    let mut extractor = TextFrameExtractor::new(
        model_path,
        output_dir,
        MASK_CHANGE_THRESHOLD_PERCENT_RUST,
        MIN_CHANGE_DURATION_MS_RUST,
        args.use_cpu,
    )?;

    extractor.clear_output_directory()?;

    // Clean up old SRT files if they exist (from Python script or previous runs)
    let raw_srt_output_file = Path::new("subtitles_raw.srt");
    let final_srt_output_file = Path::new("subtitles_final.srt");
    if raw_srt_output_file.exists() {
        fs::remove_file(raw_srt_output_file)?;
    }
    if final_srt_output_file.exists() {
        fs::remove_file(final_srt_output_file)?;
    }

    println!("Procesando video (Rust): {}", video_file_path_str);
    let start_time_extraction = std::time::Instant::now();

    extractor.process_video_file(video_file_path, display_frames)?; // display_frames = false for now

    println!(
        "Extracción de frames (Rust) tomó: {:.2} segundos.",
        start_time_extraction.elapsed().as_secs_f64()
    );

    println!("\n--- Proceso de Extracción de Frames (Rust) Completo ---");
    println!(
        "Frames extraídos en (Rust): {:?}",
        fs::canonicalize(output_dir)?
    );
    println!(
        "Tiempo total del proceso (Rust): {:.2} segundos.",
        start_time_total.elapsed().as_secs_f64()
    );

    Ok(())
}
