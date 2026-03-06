use std::cell::UnsafeCell;
use std::sync::{
    atomic::{
        AtomicUsize,
        Ordering
    }
};

const INITIAL_BUFFER_CAPACITY: usize = 256;

struct Buffer<T>{
    data: Vec<UnsafeCell<Option<T>>>,
    mask: usize
}

impl<T> Buffer<T>{
    fn new(cap: usize) -> Self{
        let data = (0..cap).map(|_| UnsafeCell::new(None)).collect();
        Self { data, mask: cap - 1 }
    }

    fn read(&self, idx: usize) -> Option<T>{
        unsafe {
            (*self.data[idx & self.mask].get()).take()
        }
    }

    fn write(&self, idx: usize, val: T){
        unsafe {
            *self.data[idx & self.mask].get() = Some(val);
        }
    }
}

pub struct WorkStealingDeque<T>{
    buffer: Buffer<T>,
    top: AtomicUsize,
    bottom: AtomicUsize,
}

impl<T> WorkStealingDeque<T>{
    fn new() -> Self{
        Self { 
            buffer: Buffer::new(INITIAL_BUFFER_CAPACITY),
            top: AtomicUsize::new(0), 
            bottom: AtomicUsize::new(0) }
    }

    fn with_capacity(n: usize) -> Self{
        Self { buffer: Buffer::new(n), top: AtomicUsize::new(0), bottom: AtomicUsize::new(0) }
    }

    pub fn push(&self, val: T){
        let b = self.bottom.load(Ordering::Relaxed);
        unsafe {self.buffer.write(b, val);}

        self.bottom.store(b+1, Ordering::Release);
    }

    pub fn pop_back(&self) -> Option<T>{
        let b = self.bottom.load(Ordering::Relaxed);
        self.bottom.store(b, Ordering::Relaxed);

        let t = self.top.load(Ordering::Acquire);
        if b < t{
            self.bottom.store(b+1, Ordering::Relaxed);
            return None;
        }

        let var = self.buffer.read(t);

        if b == t{
            if self.top.compare_exchange(t, t+1, Ordering::AcqRel, Ordering::Relaxed).is_err(){
                return None;
            }else{
                return var;
            }
        }
        return var;
    }
}