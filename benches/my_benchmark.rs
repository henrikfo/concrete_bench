#![allow(dead_code, unused_variables, unused_variables, dead_code, unused_mut, unused_imports)]
use concrete::*;
use std::time::{Duration, Instant};
use criterion::*;
use std::path::Path;

use TFHE_concurrency::*;

// --- LWE --- //

fn addition(ct1: &LWE, ct_add: &LWE) -> (){
    let ct_tmp = ct1.add_with_padding(&ct_add).unwrap();
return;}

fn real_addition(ct1: &LWE, msg: &f64) -> (){
    let ct_tmp = ct1.add_constant_dynamic_encoder(*msg).unwrap();
return;}

fn int_mult(mut ct2: &LWE) -> (){
    let ct_tmp = ct2.mul_constant_static_encoder(1).unwrap();
return;}

fn real_mult(ct3: &LWE, prec: &usize) -> (){
    let mut ct_tmp = ct3.clone();
    ct_tmp.mul_constant_with_padding_inplace(1., 1., *prec).unwrap();
return;}

fn encrypt(sk: &LWESecretKey, msg: f64, enc: &Encoder) -> (){
    let ct = LWE::encode_encrypt(&sk, msg, &enc).unwrap();
return;}

fn decrypt(sk: &LWESecretKey, ct1: &LWE) -> (){
    let msg = ct1.decrypt_decode(sk).unwrap();
return;}

fn keyswitching(ct: &LWE, ksk: &LWEKSK, enc: &Encoder) -> (){
    let ct_tmp = ct.keyswitch(&ksk).unwrap();
return;}

fn bootstrap_w_func(ct: &LWE, bsk: &LWEBSK, enc: &Encoder) -> (){
    let ct_tmp = ct.bootstrap_with_function(&bsk, |x| x, &enc).unwrap();
return;}

fn ciphertext_mult(ct: &LWE, ct_m: &LWE, bsk: &LWEBSK) -> (){
    let ct_tmp = ct.mul_from_bootstrap(&ct_m, &bsk).unwrap();
return;}

fn clone_vec(ct: &Vec<LWE>) -> (){
    let y = ct.clone();
return;}

fn clone_bsk(bsk: &LWEBSK) -> (){
    let new_bsk = bsk.clone();
return;}

fn sum_list_lwe(y: &mut Vec<LWE>) ->(){
    //let mut y = x.clone();
    let n = (y.len() as f32).log2() as usize;

    for _ in 0..n{
        let mut tmp = vec![];
        
        for j in 0..( y.len() / 2 ){

            tmp.push( y[2*j].add_with_padding(&y[2*j+1]).unwrap() );
        }
        *y = tmp;
    }
    
    let padd = y[0].encoder.nb_bit_padding - 1;
    y[0].remove_padding_inplace(padd).unwrap();

return;}

fn add_0(x: f64) -> f64{
    x+0.
}
fn parallell_boot(x: &Vec<LWE>, tfhe: &Tfheconcurrency) -> (){
    let _ = tfhe.para_boot(x.clone(), add_0);
}


// --- VECTOR LWE --- //
fn real_addition_vec(ct1: &VectorLWE, msg: &Vec<f64>) -> (){
    let ct_tmp = ct1.add_constant_dynamic_encoder(msg).unwrap();
return;}

fn addition_vec(ct1: &VectorLWE, ct_add: &VectorLWE) -> (){
    let ct_tmp = ct1.add_with_padding(&ct_add).unwrap();
return;}

fn int_mult_vec(mut ct2: &VectorLWE) -> (){
    let ct_tmp = ct2.mul_constant_static_encoder(&vec![1; 10]).unwrap();
return;}

fn real_mult_vec(ct3: &VectorLWE, prec: &usize) -> (){
    let ct_tmp = ct3.clone();
    let c = ct_tmp.mul_constant_with_padding(&vec![1.; 10], 1., *prec).unwrap();
return;}

fn sum_vec(ct: &VectorLWE) -> (){
    let ct_sum = ct.sum_with_new_min(0.).unwrap();
return;}

fn encrypt_vec(sk: &LWESecretKey, msg: &Vec<f64>, enc: &Encoder) -> (){
    let ct_vec = VectorLWE::encode_encrypt(&sk, &msg, &enc).unwrap();
return;}

fn decrypt_vec(sk: &LWESecretKey, ct: &VectorLWE) -> (){
    let msg = ct.decrypt_decode(sk).unwrap();
return;}

fn keyswitching_vec(ct: &VectorLWE, ksk: &LWEKSK, enc: &Encoder) -> (){
    let ct_tmp = ct.keyswitch(&ksk).unwrap();
return;}

fn bootstrap_w_func_vec(ct: &VectorLWE, bsk: &LWEBSK, enc: &Encoder) -> (){
    for i in 0..ct.nb_ciphertexts{
        let ct_tmp = ct.bootstrap_nth_with_function(&bsk, |x| x, &enc, i).unwrap();
    }
return;}

fn ciphertext_mult_vec(ct: &VectorLWE, ct_m: &VectorLWE, bsk: &LWEBSK) -> (){
    for i in 0..ct.nb_ciphertexts{
        let ct_tmp = ct.mul_from_bootstrap_nth(&ct_m, &bsk, i, i).unwrap();
    }
return;}

fn dot_prod(ct: &VectorLWE, ct_m: &VectorLWE, bsk: &LWEBSK) -> (){
    let mut ct_tmp1 = ct.clone();
    for i in 0..ct.nb_ciphertexts{
        let ct_tmp2 = ct.mul_from_bootstrap_nth(&ct_m, &bsk, i, i).unwrap();
        ct_tmp1.copy_in_nth_nth_inplace(i, &ct_tmp2, 0).unwrap();
    }
    let ct_sum = ct_tmp1.sum_with_new_min(0.).unwrap();
return;}

fn real_dot_prod(ct3: &VectorLWE, prec: &usize) -> (){
    //let mut ct_tmp1 = ct.clone();
    let c = ct3.mul_constant_with_padding(&vec![1.; 10], 1., *prec).unwrap();
    let ct_sum = c.sum_with_new_min(0.).unwrap();
return;}


fn bench_lwe(c: &mut Criterion){
    let mut lwe = c.benchmark_group("LWE");
    
    //let dims: Vec<usize> = vec![256, 512, 1024, 2048];
    //let noises: Vec<i32> = vec![-9, -19, -40, -62];
    let dims: Vec<usize> = vec![512, 1024, 2048, 4096];
    let noises: Vec<i32> = vec![-40, -40, -62, -62];
    let precisions: Vec<usize> = vec![3, 4, 5, 6];


    
    for ((dim, noise), precision) in dims.iter().zip(noises.iter()).zip(precisions.iter()){
        let id = format!("{}", dim);
        let enc = Encoder::new(-10., 10., *precision, 1).unwrap();
        let enc2 = Encoder::new(0., 10., *precision, 1).unwrap();
        let enc3 = Encoder::new(0., 10., *precision, *precision+1).unwrap();
        let enc4 = Encoder::new(0., 10., *precision, 2).unwrap();
        
        // secret keys
        let lwe_params: LWEParams = LWEParams::new(*dim, *noise);
        let rlwe_params: RLWEParams = RLWEParams{polynomial_size: *dim, dimension: 1, log2_std_dev: *noise};
        
        let sk = LWESecretKey::new(&lwe_params);   
        let sk_rlwe = RLWESecretKey::new(&rlwe_params);    
        let sk_out = sk_rlwe.to_lwe_secret_key();
        let bsk = LWEBSK::new(&sk, &sk_rlwe, 6, 6);
        let ksk = LWEKSK::new(&sk_out, &sk, 6, 6);
        

        // --- LWE --- //
        
        let msg1: f64 = 1.;
        let msg2: f64 = 0.;

        let mut ct1 = LWE::encode_encrypt(&sk, msg1, &enc).unwrap();
        let mut ct1_2 = LWE::encode_encrypt(&sk, msg1, &enc).unwrap();
        let ct_add = LWE::encode_encrypt(&sk, msg2, &enc).unwrap();
        let mut ct2 = LWE::encode_encrypt(&sk, msg1, &enc).unwrap();
        let ct3 = LWE::encode_encrypt(&sk, msg1, &enc3).unwrap();
        let ct_ksk = LWE::encode_encrypt(&sk_out, msg1, &enc2).unwrap();
        let ct4 = LWE::encode_encrypt(&sk, msg1, &enc2).unwrap();
        
        let ct5 = LWE::encode_encrypt(&sk, msg1, &enc4).unwrap();
        let ct_mult = LWE::encode_encrypt(&sk, msg1, &enc4).unwrap();  
        
        
        lwe.bench_with_input(BenchmarkId::new("Real Addition", &id), &(ct1, msg2), |b, (ct1, msg2)| {
            b.iter(|| real_addition(&ct1, &msg2));
        });
        
        lwe.bench_with_input(BenchmarkId::new("Addition", &id), &(ct1_2, ct_add), |b, (ct1_2, ct_add)| {
            b.iter(|| addition(&ct1_2, &ct_add));
        });

        lwe.bench_with_input(BenchmarkId::new("Interger Multiplication", &id), &ct2, |b, ct2| {
            b.iter(|| int_mult(&ct2));
        });
        
        lwe.bench_with_input(BenchmarkId::new("Real Multiplication", &id), &(&ct3, &precision), |b, (ct3, precision)| {
            b.iter(|| real_mult(&ct3, &precision));
        });
        
        lwe.bench_with_input(BenchmarkId::new("Encrypt", &id), &(&sk, &msg1, &enc2), |b, (sk, msg_vec, enc2)| {
            b.iter(|| encrypt(&sk, msg1, &enc2));
        });
        
        lwe.bench_with_input(BenchmarkId::new("Decrypt", &id), &(&sk, &ct2), |b, (sk, ct2)| {
            b.iter(|| decrypt(&sk, &ct2));
        });
        
         lwe.bench_with_input(BenchmarkId::new("Keyswitch", &id), &(&ct_ksk, &ksk, &enc2), |b, (ct_ksk, ksk, enc2)| {
            b.iter(|| keyswitching(&ct_ksk, &ksk, &enc2));
        });
        
         lwe.bench_with_input(BenchmarkId::new("Bootstrap_w_func", &id), &(&ct4, &bsk, &enc2), |b, (ct4, bsk, enc2)| {
            b.iter(|| bootstrap_w_func(&ct4, &bsk, &enc2));
        });
        
         lwe.bench_with_input(BenchmarkId::new("Ciphertext mult", &id), &(ct5, ct_mult, &bsk), |b, (ct5, ct_mult, bsk)| {
            b.iter(|| ciphertext_mult(&ct5, &ct_mult, &bsk));
        });
            
    }
}
        // --- VECTOR LWE --- //
fn bench_vectorlwe(c: &mut Criterion){        
        let mut vlwe = c.benchmark_group("VectorLWE_10");
    
        //let dims: Vec<usize> = vec![256, 512, 1024];
        //let noises: Vec<i32> = vec![-40, -62, -62];
    
        let dims: Vec<usize> = vec![512, 1024, 2048];
        let noises: Vec<i32> = vec![-40, -40, -62];
    
        let precisions: Vec<usize> = vec![3, 4, 5];


        for ((dim, noise), precision) in dims.iter().zip(noises.iter()).zip(precisions.iter()){
            //if dim > &1024{ 
            //}
            let id = format!("{}", dim);
            let enc = Encoder::new(-10., 10., *precision, 1).unwrap();
            let enc2 = Encoder::new(0., 10., *precision, 1).unwrap();
            let enc3 = Encoder::new(0., 10., *precision, *precision+1).unwrap();
            let enc4 = Encoder::new(0., 10., *precision, 2).unwrap();

            // secret keys
            let lwe_params: LWEParams = LWEParams::new(*dim, *noise);
            let rlwe_params: RLWEParams = RLWEParams{polynomial_size: *dim, dimension: 1, log2_std_dev: *noise};
            
            let sk = LWESecretKey::new(&lwe_params);   
            let sk_rlwe = RLWESecretKey::new(&rlwe_params);    
            let sk_out = sk_rlwe.to_lwe_secret_key();
            let bsk = LWEBSK::new(&sk, &sk_rlwe, 5, 5);
            let ksk = LWEKSK::new(&sk_out, &sk, 5, 5);
      
        
            let msg1_vec: Vec<f64> = vec![1.; 10];
            let msg2_vec: Vec<f64> = vec![0.; 10];
            let msg_vec: Vec<f64> = vec![0.; 10];

            let mut ct1 = VectorLWE::encode_encrypt(&sk, &msg1_vec, &enc).unwrap();
            let mut ct1_2 = VectorLWE::encode_encrypt(&sk, &msg1_vec, &enc).unwrap();
            let ct_add = VectorLWE::encode_encrypt(&sk, &msg2_vec, &enc).unwrap();
            let mut ct2 = VectorLWE::encode_encrypt(&sk, &msg1_vec, &enc).unwrap();
            let ct3 = VectorLWE::encode_encrypt(&sk, &msg1_vec, &enc3).unwrap();
            let ct_vec = VectorLWE::encode_encrypt(&sk, &msg_vec, &enc2).unwrap();
            let ct_ksk = VectorLWE::encode_encrypt(&sk_out, &msg1_vec, &enc2).unwrap();
            let ct4 = VectorLWE::encode_encrypt(&sk, &msg1_vec, &enc2).unwrap();   
            
            let ct5 = VectorLWE::encode_encrypt(&sk, &msg1_vec, &enc4).unwrap(); 
            let ct_mult = VectorLWE::encode_encrypt(&sk, &msg1_vec, &enc4).unwrap();

        
        vlwe.sample_size(100);
        vlwe.bench_with_input(BenchmarkId::new("Real Addition", &id), &(ct1_2, msg2_vec), |b, (ct1_2, msg2_vec)| {
            b.iter(|| real_addition_vec(&ct1_2, &msg2_vec));
        });
            
        vlwe.sample_size(100);
        vlwe.bench_with_input(BenchmarkId::new("Addition", &id), &(ct1, ct_add), |b, (ct1, ct_add)| {
            b.iter(|| addition_vec(&ct1, &ct_add));
        });
        
        vlwe.sample_size(100);
        vlwe.bench_with_input(BenchmarkId::new("Interger Multiplication", &id), &ct2, |b, ct2| {
            b.iter(|| int_mult_vec(&ct2));
        });
            
        vlwe.sample_size(100);
        vlwe.bench_with_input(BenchmarkId::new("Real Multiplication", &id), &(&ct2, &precision), |b, (ct2, precision)| {
            b.iter(|| real_mult_vec(&ct3, &precision));
        });
        
        vlwe.sample_size(100);
        vlwe.bench_with_input(BenchmarkId::new("Sum ciphertext vector", &id), &ct_vec, |b, ct_vec| {
            b.iter(|| sum_vec(&ct_vec));
        });

        vlwe.sample_size(100);
        vlwe.bench_with_input(BenchmarkId::new("Encrypt vector", &id), &(&sk, &msg_vec, &enc2), |b, (sk, msg_vec, enc2)| {
            b.iter(|| encrypt_vec(&sk, &msg_vec, &enc2));
        });
            
        vlwe.sample_size(100);
        vlwe.bench_with_input(BenchmarkId::new("Decrypt vector", &id), &(&sk, &ct_vec), |b, (sk, ct_vec)| {
            b.iter(|| decrypt_vec(&sk, &ct_vec));
        });
        
        vlwe.sample_size(100);
        vlwe.bench_with_input(BenchmarkId::new("Keyswitch", &id), &(&ct_ksk, &ksk, &enc2), |b, (ct_ksk, ksk, enc2)| {
            b.iter(|| keyswitching_vec(&ct_ksk, &ksk, &enc2));
        });
        
        vlwe.sample_size(100);
        vlwe.bench_with_input(BenchmarkId::new("Bootstrap_w_func", &id), &(&ct4, &bsk, &enc2), |b, (ct4, bsk, enc2)| {
            b.iter(|| bootstrap_w_func_vec(&ct4, &bsk, &enc2));
        });
        
        vlwe.sample_size(100);
        vlwe.bench_with_input(BenchmarkId::new("Ciphertext mult", &id), &(&ct5, &ct_mult, &bsk), |b, (ct5, ct_mult, bsk)| {
            b.iter(|| ciphertext_mult_vec(&ct5, &ct_mult, &bsk));
        });
        
        vlwe.sample_size(100);
        vlwe.bench_with_input(BenchmarkId::new("Dot prod", &id), &(&ct5, &ct_mult, &bsk), |b, (ct5, ct_mult, bsk)| {
            b.iter(|| dot_prod(&ct5, &ct_mult, &bsk));
        });
            
         vlwe.sample_size(100);
        vlwe.bench_with_input(BenchmarkId::new("Real dot prod", &id), &(&ct3, &precision), |b, (ct3, precision)| {
            b.iter(|| real_dot_prod(&ct3, &precision));
        });
        
    }

}

fn bench_teewondee(c: &mut Criterion){
    
    let mut lwe = c.benchmark_group("teewondee");
    
    /*
    let dims: Vec<usize> = vec![1024, 2048];
    let noises: Vec<i32> = vec![-40, -62];
    let precisions: Vec<usize> = vec![4, 5];
    */
    
    let dims: Vec<usize> = vec![512, 1024, 2048, 4096];
    let noises: Vec<i32> = vec![-19, -40, -62, -62];
    let precisions: Vec<usize> = vec![3, 4, 5, 6];
    
    
    for ((dim, noise), precision) in dims.iter().zip(noises.iter()).zip(precisions.iter()){
        let id = format!("{}", dim);
        let enc = Encoder::new(-10., 10., *precision, 1).unwrap();
        let enc2 = Encoder::new(0., 10., *precision, 1).unwrap();
        let enc3 = Encoder::new(0., 10., *precision, *precision+1).unwrap();
        let enc4 = Encoder::new(0., 10., *precision, 13).unwrap();
        
        // secret keys
        let lwe_params: LWEParams = LWEParams::new(750, -29);
        let rlwe_params: RLWEParams = RLWEParams{polynomial_size: *dim, dimension: 1, log2_std_dev: *noise};
        
        let sk = LWESecretKey::new(&lwe_params);   
        let sk_rlwe = RLWESecretKey::new(&rlwe_params);    
        let sk_out = sk_rlwe.to_lwe_secret_key();
        let bsk = LWEBSK::new(&sk, &sk_rlwe, 6, 6);
        let ksk = LWEKSK::new(&sk_out, &sk, 6, 6);
        let ksk_clone = LWEKSK::new(&sk_out, &sk, 6, 6);
        
        //let tfhe = Tfheconcurrency::new(sk.clone(), sk_out.clone(), bsk.clone(), ksk_clone, enc4.clone(), 8, false);
        
        //println!("{}", tfhe.max_threads);
        // --- LWE --- //
        
        let msg1: f64 = 1.;
        let msg2: f64 = 0.;

        let mut ct1 = LWE::encode_encrypt(&sk_out, msg1, &enc).unwrap();
        let mut ct1_2 = LWE::encode_encrypt(&sk_out, msg1, &enc).unwrap();
        let ct_add = LWE::encode_encrypt(&sk_out, msg2, &enc).unwrap();
        let mut ct2 = LWE::encode_encrypt(&sk_out, msg1, &enc).unwrap();
        let ct3 = LWE::encode_encrypt(&sk, msg1, &enc3).unwrap();
        let ct_ksk = LWE::encode_encrypt(&sk_out, msg1, &enc2).unwrap();
        let ct4 = LWE::encode_encrypt(&sk, msg1, &enc2).unwrap();
        
        let vec_lwe = LWE::encode_encrypt(&sk, msg1, &enc4).unwrap();
        let mut ct_4096 = vec![vec_lwe; 4096];
        
        
        lwe.bench_with_input(BenchmarkId::new("Real Addition", &id), &(ct1, msg2), |b, (ct1, msg2)| {
            b.iter(|| real_addition(&ct1, &msg2));
        });
        
        
        lwe.bench_with_input(BenchmarkId::new("Addition", &id), &(ct1_2, ct_add), |b, (ct1_2, ct_add)| {
            b.iter(|| addition(&ct1_2, &ct_add));
        });

        
        lwe.bench_with_input(BenchmarkId::new("Interger Multiplication", &id), &ct2, |b, ct2| {
            b.iter(|| int_mult(&ct2));
        });
        
        lwe.bench_with_input(BenchmarkId::new("Real Multiplication", &id), &(&ct3, &precision), |b, (ct3, precision)| {
            b.iter(|| real_mult(&ct3, &precision));
        });
        
        
        lwe.bench_with_input(BenchmarkId::new("Encrypt", &id), &(&sk, &msg1, &enc2), |b, (sk, msg_vec, enc2)| {
            b.iter(|| encrypt(&sk_out, msg1, &enc2));
        });
        
        lwe.bench_with_input(BenchmarkId::new("Decrypt", &id), &(&sk, &ct2), |b, (sk, ct2)| {
            b.iter(|| decrypt(&sk_out, &ct2));
        });
                
        
        lwe.bench_with_input(BenchmarkId::new("Keyswitch", &id), &(&ct_ksk, &ksk, &enc2), |b, (ct_ksk, ksk, enc2)| {
            b.iter(|| keyswitching(&ct_ksk, &ksk, &enc2));
        });
        
        lwe.bench_with_input(BenchmarkId::new("Bootstrap_w_func", &id), &(&ct4, &bsk, &enc2), |b, (ct4, bsk, enc2)| {
            b.iter(|| bootstrap_w_func(&ct4, &bsk, &enc2));
        });
        
        lwe.bench_with_input(BenchmarkId::new("Sum 4096", &id), &ct_4096, |b, ct_4096| {
            b.iter(|| sum_list_lwe(&mut ct_4096.clone()));
        });
        
        
        lwe.bench_with_input(BenchmarkId::new("Clone Vec of LWE", &id), &ct_4096, |b, ct_4096|{
            b.iter(|| clone_vec(&ct_4096));
        });
        
        
        lwe.bench_with_input(BenchmarkId::new("Clone BSK", &id), &bsk, |b, bsk|{
            b.iter(|| clone_bsk(&bsk));
        });
        
        /*
        if dim < &4000{
            lwe.sample_size(10);
            lwe.bench_with_input(BenchmarkId::new("Parallell bootstrapping", &id), &(&ct_4096, &tfhe), |b, (ct_4096, tfhe)| {
            b.iter(|| parallell_boot(&ct_4096, &tfhe));
            });
        }*/
        
            
    }
}

//criterion_group!(benches, bench_lwe, bench_vectorlwe);
criterion_group!(benches, bench_lwe);
//criterion_group!(benches, bench_vectorlwe);
//criterion_group!(benches, bench_teewondee);
//criterion_group!(benches, bench_lwe, bench_vectorlwe, bench_teewondee);
criterion_main!(benches);
