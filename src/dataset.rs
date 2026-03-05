use crate::ml::{Tensor, TensorData};

pub trait Stackable: Sized {
    type Output;
    fn stack(batch: Vec<Self>) -> Self::Output;
}

impl Stackable for Tensor {
    type Output = Tensor;
    fn stack(batch: Vec<Self>) -> Self::Output {
        assert!(!batch.is_empty(), "batch length is 0");

        let mut data = Vec::new();

        for bs in batch.windows(2) {
            let datatype_flag = match (&bs[0].data, &bs[1].data) {
                (&TensorData::F32(_), &TensorData::F32(_)) => true,
                (
                    &TensorData::I8 { data: _, scales: _ },
                    &TensorData::I8 { data: _, scales: _ },
                ) => true,
                _ => false,
            };
            assert!(
                datatype_flag,
                "data type is different, bs[0]={:#?}, bs[1]={:#?}",
                bs[0], bs[1]
            );
            let datasize_flag = bs[0].len() == bs[1].len();
            assert!(
                datasize_flag,
                "data size is different, bs[0].len()={}, bs[1].len()={}",
                bs[0].len(),
                bs[1].len()
            );
            let datashape_flag = bs[0]
                .shape
                .iter()
                .zip(bs[1].shape.iter())
                .all(|(l, r)| l == r);
            assert!(
                datashape_flag,
                "data shape is different, bs[0].shape={:#?}, bs[1].shape={:#?}",
                bs[0].shape, bs[1].shape
            );
        }

        for b in batch.iter() {
            let b_f32 = b.as_f32_slice();
            data.extend_from_slice(&b_f32);
        }

        let mut shape = vec![batch.len()];
        shape.extend_from_slice(&batch[0].shape);

        Tensor::new(data, shape)
    }
}

pub trait Dataset {
    type Item: Stackable;

    fn len(&self) -> usize;
    fn get(&self, index: usize) -> Self::Item;
}

pub struct Dataloader<D>
where
    D: Dataset,
{
    dataset: D,
    batch_size: usize,
    strict_batch_size: bool,
}

impl<D: Dataset> Dataloader<D> {
    pub fn new(dataset: D, batch_size: usize, strict_batch_size: bool) -> Self {
        Self {
            dataset,
            batch_size,
            strict_batch_size,
        }
    }

    pub fn iter_batch(&self) -> BatchIterator<'_, D> {
        let dataset_size = self.dataset.len();
        let mut indices: Vec<usize> = (0..dataset_size).collect();

        use crate::utills::rand::RngCore;
        crate::utills::rand::rng().shuffle(&mut indices);

        BatchIterator {
            dataloader: self,
            strict_batch_size: self.strict_batch_size,
            batch_size: self.batch_size,
            indices,
        }
    }

    pub fn len(&self) -> usize {
        self.dataset.len()
    }
}

pub struct BatchIterator<'a, D: Dataset> {
    dataloader: &'a Dataloader<D>,
    strict_batch_size: bool,
    batch_size: usize,
    indices: Vec<usize>,
}

impl<'a, D: Dataset> Iterator for BatchIterator<'a, D> {
    type Item = <D::Item as Stackable>::Output;

    fn next(&mut self) -> Option<Self::Item> {
        let mut v = Vec::new();
        for i in 0..self.batch_size {
            if let Some(index) = self.indices.pop() {
                let entry = self.dataloader.dataset.get(index);
                v.push(entry);
            } else {
                if i == 0 || self.strict_batch_size {
                    return None;
                }
                break;
            }
        }

        let batch = D::Item::stack(v);

        Some(batch)
    }
}

// Re-export the derive macro from stackable_derive
pub use stackable_derive::Stackable;
