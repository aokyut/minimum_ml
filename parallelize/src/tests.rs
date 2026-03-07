use crate::thread_pool::{self, tiny::{self, TINY_THREAD_POOL, TinyThreadPool}};
use std::time::Instant;

#[test]
fn bench_thread_pool(){
    use std::hint::black_box;
    let n = 100;
    let size = 1000000;
    crate::set_thread_size(16);
    let mut acums = vec![0; 12];
    let chunk_sizes: Vec<_> = (8..12).map(|a|1 << a).collect();
    let mut acum = 0;
    let f = |a: &f32|{
        let mut x = *a;
        for _ in 0..10 { // 負荷を増やす
            x = 1.0 / (2.0 + x.sin());
        }
        x
    };

    for _ in 0..n{
        for (i, chunk_size) in chunk_sizes.iter().enumerate(){
            let v: Vec<_> = (0..size).map(|a|a as f32).collect();
            let mut out = vec![0.0; size];
            
            let now = Instant::now();
            thread_pool::parallel_map_chunked(
                &*tiny::TINY_THREAD_POOL,
                &v,
                &mut out,
                f, 
                *chunk_size
            );
            let time = now.elapsed().as_nanos();
            println!("{}", black_box(out[0])); 
            acums[i] += time;
        }
        
        let v: Vec<_> = (0..size).map(|a|a as f32).collect();
        let mut out: Vec<f32> = vec![0.0; size];

        let t = Instant::now();
        for (o, i) in out.iter_mut().zip(v.iter()){
            *o = f(i)
        }
        let time = t.elapsed().as_nanos();
        println!("{}", out[0]);
        // println!("{:#?}", out);
        acum += time;
    }

    println!("{}", TINY_THREAD_POOL.workers.len());

    let single = acum/n;
    println!("[single]time:{}", acum / n);

    for (i, chunk_size) in chunk_sizes.iter().enumerate(){
        let time = acums[i];
        let per_time = time / n;
        println!("[chunk_size:{chunk_size}]time:{}[{}%]", per_time, 100 * per_time / single);
    }
}