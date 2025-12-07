use eyre::Result;
use hound::WavReader;
use std::path::Path;

pub fn read_wav<P: AsRef<Path>>(file_path: P) -> Result<(Vec<i16>, u32)> {
    let mut reader = WavReader::open(file_path.as_ref())?;
    let spec = reader.spec();
    let sample_rate = spec.sample_rate;
    let samples: Vec<i16> = reader.samples::<i16>().collect::<Result<Vec<_>, _>>()?;

    Ok((samples, sample_rate))
}
