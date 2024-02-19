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
                fwht_4(data, i as u16, dist as u16);
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
fn fwht_2(a: GfElement, b: GfElement) -> (GfElement, GfElement) {
    let sum = engine::add_mod(a, b);
    let dif = engine::sub_mod(a, b);
    (sum, dif)
}

#[inline(always)]
fn fwht_4(data: &mut [GfElement; GF_ORDER], offset: u16, dist: u16) {
    // Indices. u16 additions and multiplication to avoid bounds checks
    // on array access. (GF_ORDER == (u16::MAX+1))
    let i0 = usize::from(offset);
    let i1 = usize::from(offset + dist);
    let i2 = usize::from(offset + dist * 2);
    let i3 = usize::from(offset + dist * 3);

    let (s0, d0) = fwht_2(data[i0], data[i1]);
    let (s1, d1) = fwht_2(data[i2], data[i3]);
    let (s2, d2) = fwht_2(s0, s1);
    let (s3, d3) = fwht_2(d0, d1);

    data[i0] = s2;
    data[i1] = s3;
    data[i2] = d2;
    data[i3] = d3;
}
