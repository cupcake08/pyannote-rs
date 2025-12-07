//! Diarization Pipeline Module
//!
//! Provides high-level API for speaker diarization:
//! - Segmentation (VAD)
//! - Speaker embedding extraction
//! - Speaker clustering/identification
//! - Segment merging for same-speaker chunks

use crate::{embedding::EmbeddingExtractor, identify::EmbeddingManager, segment, session, wav};
use eyre::{Context, Result};
use std::path::Path;

/// A speech segment with speaker identification
#[derive(Debug, Clone)]
pub struct LabeledSegment {
    /// Start time in seconds
    pub start: f64,
    /// End time in seconds
    pub end: f64,
    /// Speaker ID (1-indexed, 0 = unknown)
    pub speaker_id: usize,
    /// Raw audio samples (16-bit PCM)
    pub samples: Vec<i16>,
}

/// Merge consecutive segments from the same speaker if gap is small enough.
/// This provides longer audio chunks for better transcription accuracy.
pub fn merge_same_speaker_segments(
    segments: Vec<LabeledSegment>,
    max_gap_seconds: f64,
) -> Vec<LabeledSegment> {
    if segments.is_empty() {
        return Vec::new();
    }

    let mut merged: Vec<LabeledSegment> = Vec::new();
    let mut current = segments[0].clone();

    for seg in segments.into_iter().skip(1) {
        let gap = seg.start - current.end;
        let same_speaker = seg.speaker_id == current.speaker_id;

        if same_speaker && gap <= max_gap_seconds {
            // Extend current segment
            current.end = seg.end;
            current.samples.extend(seg.samples);
        } else {
            // Save current and start new
            merged.push(current);
            current = seg;
        }
    }

    // Don't forget the last segment
    merged.push(current);
    merged
}

/// Full diarization pipeline: segment → embed → identify → merge
pub struct DiarizationPipeline {
    segmenter_session: ort::session::Session,
    embedding_extractor: EmbeddingExtractor,
    embedding_manager: EmbeddingManager,
    search_threshold: f32,
    max_gap_seconds: f64,
}

impl DiarizationPipeline {
    /// Create a new diarization pipeline
    ///
    /// # Arguments
    /// * `segmentation_model_path` - Path to segmentation ONNX model
    /// * `embedding_model_path` - Path to embedding ONNX model
    /// * `xvec_transform_path` - Optional path to xvec_transform.npz for PLDA
    /// * `plda_path` - Optional path to plda.npz for PLDA
    pub fn new<P: AsRef<Path>>(
        segmentation_model_path: P,
        embedding_model_path: P,
        xvec_transform_path: Option<P>,
        plda_path: Option<P>,
    ) -> Result<Self> {
        let segmenter_session = session::create_session(segmentation_model_path.as_ref())?;

        let embedding_extractor = match (xvec_transform_path, plda_path) {
            (Some(xvec), Some(plda)) => EmbeddingExtractor::new_with_plda(
                embedding_model_path,
                xvec,
                plda,
                128, // LDA dimension
            )?,
            _ => EmbeddingExtractor::new(embedding_model_path)?,
        };

        Ok(Self {
            segmenter_session,
            embedding_extractor,
            embedding_manager: EmbeddingManager::new(usize::MAX),
            search_threshold: 0.5,
            max_gap_seconds: 0.5,
        })
    }

    /// Set speaker search threshold (default: 0.5)
    pub fn with_search_threshold(mut self, threshold: f32) -> Self {
        self.search_threshold = threshold;
        self
    }

    /// Set max gap for merging same-speaker segments (default: 0.5s)
    pub fn with_max_gap(mut self, max_gap: f64) -> Self {
        self.max_gap_seconds = max_gap;
        self
    }

    /// Reset speaker memory (for new audio files)
    pub fn reset_speakers(&mut self) {
        self.embedding_manager = EmbeddingManager::new(usize::MAX);
    }

    /// Process audio file and return labeled, merged segments
    pub fn process<P: AsRef<Path>>(&mut self, audio_path: P) -> Result<Vec<LabeledSegment>> {
        // Read audio
        let (samples, sample_rate) = wav::read_wav(audio_path.as_ref())?;

        self.process_samples(&samples, sample_rate)
    }

    /// Process raw audio samples and return labeled, merged segments
    pub fn process_samples(
        &mut self,
        samples: &[i16],
        sample_rate: u32,
    ) -> Result<Vec<LabeledSegment>> {
        // Segment audio
        let raw_segments =
            segment::get_segments_with_session(samples, sample_rate, &mut self.segmenter_session)?;

        // Process each segment: extract embedding and identify speaker
        let mut labeled_segments: Vec<LabeledSegment> = Vec::new();

        // Minimum samples for reliable embedding (0.5 seconds at 16kHz = 8000 samples)
        const MIN_SAMPLES_FOR_EMBEDDING: usize = 8000;

        for seg_result in raw_segments {
            let seg = seg_result.context("Failed to get segment")?;

            // Skip segments that are too short for embedding extraction
            if seg.samples.len() < MIN_SAMPLES_FOR_EMBEDDING {
                continue;
            }

            // Extract embedding
            let embedding = match self.embedding_extractor.compute(&seg.samples) {
                Ok(emb) => emb,
                Err(e) => {
                    // Log and skip failed segments instead of crashing
                    eprintln!(
                        "Warning: Failed to compute embedding for segment {:.2}-{:.2}: {}",
                        seg.start, seg.end, e
                    );
                    continue;
                }
            };

            // Identify speaker (or create new one)
            let speaker_id = self
                .embedding_manager
                .search_speaker(embedding.clone(), self.search_threshold)
                .or_else(|| self.embedding_manager.search_speaker(embedding, 0.0))
                .unwrap_or(0);

            labeled_segments.push(LabeledSegment {
                start: seg.start,
                end: seg.end,
                speaker_id,
                samples: seg.samples,
            });
        }

        // Merge consecutive same-speaker segments
        Ok(merge_same_speaker_segments(
            labeled_segments,
            self.max_gap_seconds,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merge_empty() {
        let result = merge_same_speaker_segments(vec![], 0.5);
        assert!(result.is_empty());
    }

    #[test]
    fn test_merge_single() {
        let segments = vec![LabeledSegment {
            start: 0.0,
            end: 1.0,
            speaker_id: 1,
            samples: vec![1, 2, 3],
        }];
        let result = merge_same_speaker_segments(segments, 0.5);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_merge_same_speaker_small_gap() {
        let segments = vec![
            LabeledSegment {
                start: 0.0,
                end: 1.0,
                speaker_id: 1,
                samples: vec![1, 2],
            },
            LabeledSegment {
                start: 1.3, // 0.3s gap
                end: 2.0,
                speaker_id: 1,
                samples: vec![3, 4],
            },
        ];
        let result = merge_same_speaker_segments(segments, 0.5);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].start, 0.0);
        assert_eq!(result[0].end, 2.0);
        assert_eq!(result[0].samples, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_no_merge_different_speakers() {
        let segments = vec![
            LabeledSegment {
                start: 0.0,
                end: 1.0,
                speaker_id: 1,
                samples: vec![1, 2],
            },
            LabeledSegment {
                start: 1.1,
                end: 2.0,
                speaker_id: 2,
                samples: vec![3, 4],
            },
        ];
        let result = merge_same_speaker_segments(segments, 0.5);
        assert_eq!(result.len(), 2);
    }
}
