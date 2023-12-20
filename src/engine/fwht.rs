use crate::engine::{self, GfElement, GF_ORDER};

// ======================================================================
// FWHT (fast Walsh-Hadamard transform) - CRATE

#[inline(always)]
pub(crate) fn fwht(data: &mut [GfElement; GF_ORDER], truncated_size: usize) {
    // TWO LAYERS AT TIME

    let mut dist = 1;
    let mut dist4 = 4;
    while dist4 <= GF_ORDER {
        let mut r = 0;
        while r < truncated_size {
            for i in r..r + dist {
                fwht_4(&mut data[i..], dist)
            }
            r += dist4;
        }

        dist = dist4;
        dist4 <<= 2;
    }

    // FINAL ODD LAYER

    if dist < GF_ORDER {
        for i in 0..dist {
            // inlined manually as Rust doesn't like
            // `fwht_2(&mut data[i], &mut data[i + dist])`
            let sum = engine::add_mod(data[i], data[i + dist]);
            let dif = engine::sub_mod(data[i], data[i + dist]);
            data[i] = sum;
            data[i + dist] = dif;
        }
    }
}

// ======================================================================
// FWHT - PRIVATE

#[inline(always)]
fn fwht_2(a: &mut GfElement, b: &mut GfElement) {
    let sum = engine::add_mod(*a, *b);
    let dif = engine::sub_mod(*a, *b);
    *a = sum;
    *b = dif;
}

#[inline(always)]
fn fwht_4(data: &mut [GfElement], dist: usize) {
    let mut t0 = data[0];
    let mut t1 = data[dist];
    let mut t2 = data[dist * 2];
    let mut t3 = data[dist * 3];

    fwht_2(&mut t0, &mut t1);
    fwht_2(&mut t2, &mut t3);
    fwht_2(&mut t0, &mut t2);
    fwht_2(&mut t1, &mut t3);

    data[0] = t0;
    data[dist] = t1;
    data[dist * 2] = t2;
    data[dist * 3] = t3;
}
