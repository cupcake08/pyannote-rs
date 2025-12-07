#[cfg(feature = "python")]
use pyo3::prelude::*;

mod embedding;
mod identify;
mod pipeline;
mod plda;
mod segment;
mod session;
mod wav;

pub use embedding::EmbeddingExtractor;
pub use identify::EmbeddingManager;
pub use knf_rs::{compute_fbank, convert_integer_to_float_audio};
pub use pipeline::{merge_same_speaker_segments, DiarizationPipeline, LabeledSegment};
pub use plda::PLDA;
pub use segment::{get_segments, Segment};
pub use wav::read_wav;

#[cfg(feature = "python")]
#[pyclass(name = "Segment")]
struct PySegment {
    #[pyo3(get)]
    start: f64,
    #[pyo3(get)]
    end: f64,
    #[pyo3(get)]
    samples: Vec<i16>,
}

#[cfg(feature = "python")]
#[pyclass(name = "Segmenter")]
struct PySegmenter {
    session: ort::session::Session,
}

#[cfg(feature = "python")]
#[pymethods]
impl PySegmenter {
    #[new]
    fn new(model_path: String) -> PyResult<Self> {
        let session = session::create_session(&model_path).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to load model: {}",
                e
            ))
        })?;
        Ok(PySegmenter { session })
    }

    fn segment(&mut self, samples: Vec<i16>, sample_rate: u32) -> PyResult<Vec<PySegment>> {
        let segments_iter = segment::get_segments_with_session(
            &samples,
            sample_rate,
            &mut self.session,
        )
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Segmentation failed: {}", e))
        })?;

        let mut result = Vec::new();
        for seg_res in segments_iter {
            let seg = seg_res.map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Segment error: {}", e))
            })?;
            result.push(PySegment {
                start: seg.start,
                end: seg.end,
                samples: seg.samples,
            });
        }
        Ok(result)
    }
}

#[cfg(feature = "python")]
#[pyclass(name = "EmbeddingExtractor")]
struct PyEmbeddingExtractor {
    inner: EmbeddingExtractor,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyEmbeddingExtractor {
    #[new]
    fn new(model_path: String) -> PyResult<Self> {
        let inner = EmbeddingExtractor::new(&model_path).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to load embedding model: {}",
                e
            ))
        })?;
        Ok(PyEmbeddingExtractor { inner })
    }

    #[staticmethod]
    fn new_with_plda(
        model_path: String,
        xvec_transform_path: String,
        plda_path: String,
        lda_dimension: usize,
    ) -> PyResult<Self> {
        let inner = EmbeddingExtractor::new_with_plda(
            &model_path,
            &xvec_transform_path,
            &plda_path,
            lda_dimension,
        )
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to load embedding model with PLDA: {}",
                e
            ))
        })?;
        Ok(PyEmbeddingExtractor { inner })
    }

    fn compute(&mut self, samples: Vec<i16>) -> PyResult<Vec<f32>> {
        self.inner.compute(&samples).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Embedding computation failed: {}",
                e
            ))
        })
    }
}

#[cfg(feature = "python")]
#[pyclass(name = "EmbeddingManager")]
struct PyEmbeddingManager {
    inner: EmbeddingManager,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyEmbeddingManager {
    #[staticmethod]
    fn new() -> PyResult<Self> {
        Ok(PyEmbeddingManager {
            inner: EmbeddingManager::new(usize::MAX),
        })
    }

    fn search_speaker(&mut self, embedding: Vec<f32>, threshold: f32) -> PyResult<String> {
        let speaker = self
            .inner
            .search_speaker(embedding.clone(), threshold)
            .or_else(|| self.inner.search_speaker(embedding, 0.0))
            .map(|r| r.to_string())
            .unwrap_or_else(|| "?".to_string());
        Ok(speaker)
    }
}

// ============================================================================
// NEW: LabeledSegment and DiarizationPipeline Python bindings
// ============================================================================

#[cfg(feature = "python")]
#[pyclass(name = "LabeledSegment")]
#[derive(Clone)]
struct PyLabeledSegment {
    #[pyo3(get)]
    start: f64,
    #[pyo3(get)]
    end: f64,
    #[pyo3(get)]
    speaker_id: usize,
    #[pyo3(get)]
    samples: Vec<i16>,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyLabeledSegment {
    fn __repr__(&self) -> String {
        format!(
            "LabeledSegment(start={:.2}, end={:.2}, speaker={})",
            self.start, self.end, self.speaker_id
        )
    }
}

#[cfg(feature = "python")]
#[pyclass(name = "DiarizationPipeline")]
struct PyDiarizationPipeline {
    inner: DiarizationPipeline,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyDiarizationPipeline {
    /// Create a new diarization pipeline
    ///
    /// Args:
    ///     segmentation_model_path: Path to segmentation ONNX model
    ///     embedding_model_path: Path to embedding ONNX model
    ///     xvec_transform_path: Optional path to xvec_transform.npz for PLDA
    ///     plda_path: Optional path to plda.npz for PLDA
    ///     search_threshold: Speaker matching threshold (default: 0.5)
    ///     max_gap_seconds: Max gap for merging same-speaker segments (default: 0.5)
    #[new]
    #[pyo3(signature = (segmentation_model_path, embedding_model_path, xvec_transform_path=None, plda_path=None, search_threshold=0.5, max_gap_seconds=0.5))]
    fn new(
        segmentation_model_path: String,
        embedding_model_path: String,
        xvec_transform_path: Option<String>,
        plda_path: Option<String>,
        search_threshold: f32,
        max_gap_seconds: f64,
    ) -> PyResult<Self> {
        let inner = match (&xvec_transform_path, &plda_path) {
            (Some(xvec), Some(plda)) => DiarizationPipeline::new(
                &segmentation_model_path,
                &embedding_model_path,
                Some(xvec),
                Some(plda),
            ),
            _ => DiarizationPipeline::new::<&str>(
                &segmentation_model_path,
                &embedding_model_path,
                None,
                None,
            ),
        }
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to create pipeline: {}",
                e
            ))
        })?
        .with_search_threshold(search_threshold)
        .with_max_gap(max_gap_seconds);

        Ok(PyDiarizationPipeline { inner })
    }

    /// Process an audio file and return labeled, merged segments
    fn process(&mut self, audio_path: String) -> PyResult<Vec<PyLabeledSegment>> {
        let segments = self.inner.process(&audio_path).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Processing failed: {}", e))
        })?;

        Ok(segments
            .into_iter()
            .map(|s| PyLabeledSegment {
                start: s.start,
                end: s.end,
                speaker_id: s.speaker_id,
                samples: s.samples,
            })
            .collect())
    }

    /// Process raw audio samples and return labeled, merged segments
    fn process_samples(
        &mut self,
        samples: Vec<i16>,
        sample_rate: u32,
    ) -> PyResult<Vec<PyLabeledSegment>> {
        let segments = self
            .inner
            .process_samples(&samples, sample_rate)
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Processing failed: {}",
                    e
                ))
            })?;

        Ok(segments
            .into_iter()
            .map(|s| PyLabeledSegment {
                start: s.start,
                end: s.end,
                speaker_id: s.speaker_id,
                samples: s.samples,
            })
            .collect())
    }

    /// Reset speaker memory (for new audio files with different speakers)
    fn reset_speakers(&mut self) {
        self.inner.reset_speakers();
    }
}

#[cfg(feature = "python")]
#[pyfunction]
fn read_wav_file(path: String) -> PyResult<(Vec<i16>, u32)> {
    read_wav(&path).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to read wav: {}", e))
    })
}

#[cfg(feature = "python")]
#[pymodule]
fn pyannote_rs(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Original classes
    m.add_class::<PySegment>()?;
    m.add_class::<PySegmenter>()?;
    m.add_class::<PyEmbeddingExtractor>()?;
    m.add_class::<PyEmbeddingManager>()?;
    // New pipeline classes
    m.add_class::<PyLabeledSegment>()?;
    m.add_class::<PyDiarizationPipeline>()?;
    // Functions
    m.add_function(wrap_pyfunction!(read_wav_file, m)?)?;
    Ok(())
}
