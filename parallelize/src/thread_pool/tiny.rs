use std::sync::atomic::AtomicI32;
use std::thread::{self, Thread};
use std::sync::{
    Arc,
    Mutex,
    mpsc,
    LazyLock,
    atomic::{
        AtomicUsize,
        AtomicBool,
        Ordering
    }
};
use crate::{get_thread_size, thread_pool::ThreadPool};

pub const TINY_THREAD_POOL: LazyLock<TinyThreadPool> =
    LazyLock::new(|| {
        let count = get_thread_size();
        TinyThreadPool::new(count)
    });
static TINY_THREAD_POOL_END_FLAG: AtomicBool = AtomicBool::new(false);

type Job = Box<dyn FnOnce() + Send + 'static>;

pub fn setup(){
    TINY_THREAD_POOL.spawn(||{
        let _ = 1;
    });
}

pub struct TinyThreadPool{
    pub workers: Vec<thread::JoinHandle<()>>,
    sender: mpsc::Sender<Job>,
    sent: Arc<AtomicUsize>,
    completed: Arc<AtomicUsize>,
}

impl TinyThreadPool{
    pub fn new(size: usize) -> Self{
        let (sender, reciever) = mpsc::channel::<Job>();
        let receiver = Arc::new(Mutex::new(reciever));

        let mut workers = Vec::new();
        let sent = Arc::new(AtomicUsize::new(0));
        let completed = Arc::new(AtomicUsize::new(0));

        for i in 0..size{
            let r = receiver.clone();
            let c = completed.clone();
            let thread = thread::spawn(move ||{
                loop {
                    // ブロックでguardをすぐdrop → Mutex解放
                    let job = {
                        let g = r.lock().unwrap();
                        g.recv()
                    };

                    match job {
                        Ok(job) => {
                            job();
                            // println!("message-{i}");
                            c.fetch_add(1, Ordering::Release);
                        },
                        Err(_) => break, // ← senderがdropされたら終了
                    }
                    if TINY_THREAD_POOL_END_FLAG.load(Ordering::Relaxed){
                        break;
                    }
                }
            });

            workers.push(thread);
        }

        Self { workers: workers, sender: sender, sent, completed }
    }
}

impl ThreadPool for TinyThreadPool{
    fn sweap(&mut self) {
        TINY_THREAD_POOL_END_FLAG.store(false, Ordering::Relaxed);
    }
    fn spawn<F>(&self, f: F)
        where
            F: FnOnce() + Send + 'static {
        self.sent.fetch_add(1, Ordering::Relaxed);
        self.sender.send(Box::new(f)).unwrap();
    }
    fn join(&self) {
        loop{
            let s = self.sent.load(Ordering::Relaxed);
            let c = self.completed.load(Ordering::Acquire);
            if s == c{
                break;
            }
            thread::yield_now(); 
        }
    }
}