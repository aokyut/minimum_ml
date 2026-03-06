pub mod tiny;
pub mod work_stealing;

use std::sync::{Arc, Barrier};
use std::ops::Deref;
use std::mem::ManuallyDrop;

pub trait ThreadPool{
    fn sweap(&mut self);
    fn spawn<F>(&self, f: F)
    where
        F: FnOnce() + Send + 'static;
    fn join(&self);
}

struct SendMutPtr<T>(*mut T);
unsafe impl<T: Send> Send for SendMutPtr<T> {}
unsafe impl<T: Send> Sync for SendMutPtr<T> {}

impl<T> SendMutPtr<T>{
    fn get(self) -> *mut T{
        self.0
    }
}

struct SendConstPtr<T>(*const T);
unsafe impl<T: Send> Send for SendConstPtr<T>{}
unsafe impl<T: Send> Sync for SendConstPtr<T>{}

impl<T> SendConstPtr<T>{
    fn get(self) -> *const T{
        self.0
    }
}

pub fn parallel_map_chunked<P, I, F, O>(
    pool: &P,
    in_vec: &[I],
    out_vec: &mut Vec<O>,
    f: F,
    chunk_size: usize,
) where
    P: ThreadPool,
    F: Fn(&I) -> O + Send + Sync + 'static,
    I: Send + 'static,
    O: Send + 'static,
{
    let f = Arc::new(f);

    let in_chunks: Vec<(SendConstPtr<I>, usize)> = in_vec
        .chunks(chunk_size)
        .map(|c| (SendConstPtr(c.as_ptr()), c.len()))
        .collect();

    let out_chunks: Vec<(SendMutPtr<O>, usize)> = out_vec
        .chunks_mut(chunk_size)
        .map(|c| (SendMutPtr(c.as_mut_ptr()), c.len()))
        .collect();

    for (i, ((in_ptr, in_len), (out_ptr, out_len))) in
        in_chunks.into_iter().zip(out_chunks.into_iter()).enumerate()
    {
        let f = Arc::clone(&f);

        pool.spawn(move || {
            let in_slice = unsafe { std::slice::from_raw_parts(in_ptr.get(), in_len) };
            let out_slice = unsafe { std::slice::from_raw_parts_mut(out_ptr.get(), out_len) };

            for (x, a) in out_slice.iter_mut().zip(in_slice.iter()) {
                *x = f(a);
            }
            // println!("end:{i}");
        });
    }
    pool.join();
}