pub mod thread_pool;
#[cfg(test)]
mod tests;

use std::thread;
use std::sync::{
    atomic::{
        AtomicUsize,
        Ordering
}};

static THREAD_COUNT: AtomicUsize = AtomicUsize::new(0);

pub fn get_thread_size() -> usize{
    let t = THREAD_COUNT.load(Ordering::Acquire);
    if t == 0{
        match thread::available_parallelism(){
            Ok(availables) => {
                availables.get()
            },
            Err(_) => {
                println!("Failed to get the number of available parallel processes");
                1
            }
        }
    }else{
        t
    }
}

pub fn set_thread_size(n: usize){
    THREAD_COUNT.store(n, Ordering::Relaxed);
}
