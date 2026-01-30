pub mod dataset;
pub mod ml;
pub mod quantize;
pub mod utills;

#[cfg(test)]
mod test;

pub fn add(left: u64, right: u64) -> u64 {
    left + right
}
