use crate::engine::{self, GfElement, GF_ORDER};

// ======================================================================
// FWHT (fast Walsh-Hadamard transform) - CRATE

/// Decimation in time (DIT) Fast Walsh-Hadamard Transform.
/// `m_truncated`: Number of non-zero elements in `data` (at the front).
#[inline(always)]
pub(crate) fn fwht(data: &mut [GfElement; GF_ORDER], m_truncated: usize) {
    // Note to self: fwht_8 is slightly faster on x86 (AMD Ryzen 5 3600),
    // but slower on ARM (Apple silicon M1).
    // fwht_16 is always slower. See branch: AndersTrier/FWHT_8_and_16

    if m_truncated >= GF_ORDER {
        return fwht_16_full(data);
    }

    let mut dist = 1;
    let mut dist4 = 4;
    while dist4 <= GF_ORDER {
        for r in (0..m_truncated).step_by(dist4) {
            for offset in r..r + dist {
                fwht_4(data, offset as u16, dist as u16);
            }
        }

        dist = dist4;
        dist4 <<= 2;
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

#[inline(always)]
fn fwht_8_full(data: &mut [GfElement; GF_ORDER]) {
    fwht_8_truncated(data, GF_ORDER)
}

#[inline(always)]
fn fwht_8_truncated(data: &mut [GfElement; GF_ORDER], m_truncated: usize) {
    let mut dist = 1;
    let mut dist8 = 8;
    while dist8 <= GF_ORDER {
        for r in (0..m_truncated).step_by(dist8) {
            for offset in r..r + dist {
                fwht_8(data, offset as u16, dist as u16);
            }
        }

        dist = dist8;
        dist8 <<= 3;
    }

    for i in 0..32768 {
        let (s0, d0) = fwht_2(data[i], data[i + dist]);
        data[i] = s0;
        data[i + dist] = d0;
    }
}

#[inline(always)]
fn fwht_8(data: &mut [GfElement; GF_ORDER], offset: u16, dist: u16) {
    let t0 = usize::from(offset);
    let t1 = usize::from(offset + dist);
    let t2 = usize::from(offset + dist * 2);
    let t3 = usize::from(offset + dist * 3);
    let t4 = usize::from(offset + dist * 4);
    let t5 = usize::from(offset + dist * 5);
    let t6 = usize::from(offset + dist * 6);
    let t7 = usize::from(offset + dist * 7);

    let (s0, d0) = fwht_2(data[t0], data[t1]);
    let (s1, d1) = fwht_2(data[t2], data[t3]);
    let (s2, d2) = fwht_2(data[t4], data[t5]);
    let (s3, d3) = fwht_2(data[t6], data[t7]);

    let (s4, d4) = fwht_2(s0, s1);
    let (s5, d5) = fwht_2(s2, s3);
    let (s6, d6) = fwht_2(d0, d1);
    let (s7, d7) = fwht_2(d2, d3);

    let (s8, d8) = fwht_2(s4, s5);
    let (s9, d9) = fwht_2(s6, s7);
    let (s10, d10) = fwht_2(d4, d5);
    let (s11, d11) = fwht_2(d6, d7);

    data[t0] = s8;
    data[t1] = s9;
    data[t2] = s10;
    data[t3] = s11;
    data[t4] = d8;
    data[t5] = d9;
    data[t6] = d10;
    data[t7] = d11;
}

#[inline(always)]
fn fwht_16_full(data: &mut [GfElement; GF_ORDER]) {
    let mut dist = 1;
    let mut dist16 = 16;
    while dist16 <= GF_ORDER {
        for r in (0..GF_ORDER).step_by(dist16) {
            for offset in r..r + dist {
                fwht_16(data, offset as u16, dist as u16);
            }
        }

        dist = dist16;
        dist16 <<= 4;
    }
}

#[inline(always)]
fn fwht_16(data: &mut [GfElement; GF_ORDER], offset: u16, dist: u16) {
    let t0 = usize::from(offset);
    let t1 = usize::from(offset + dist);
    let t2 = usize::from(offset + dist * 2);
    let t3 = usize::from(offset + dist * 3);
    let t4 = usize::from(offset + dist * 4);
    let t5 = usize::from(offset + dist * 5);
    let t6 = usize::from(offset + dist * 6);
    let t7 = usize::from(offset + dist * 7);
    let t8 = usize::from(offset + dist * 8);
    let t9 = usize::from(offset + dist * 9);
    let t10 = usize::from(offset + dist * 10);
    let t11 = usize::from(offset + dist * 11);
    let t12 = usize::from(offset + dist * 12);
    let t13 = usize::from(offset + dist * 13);
    let t14 = usize::from(offset + dist * 14);
    let t15 = usize::from(offset + dist * 15);

    let (s0, d0) = fwht_2(data[t0], data[t1]);
    let (s1, d1) = fwht_2(data[t2], data[t3]);
    let (s2, d2) = fwht_2(data[t4], data[t5]);
    let (s3, d3) = fwht_2(data[t6], data[t7]);
    let (s4, d4) = fwht_2(data[t8], data[t9]);
    let (s5, d5) = fwht_2(data[t10], data[t11]);
    let (s6, d6) = fwht_2(data[t12], data[t13]);
    let (s7, d7) = fwht_2(data[t14], data[t15]);

    let (s8, d8) = fwht_2(s0, s1);
    let (s9, d9) = fwht_2(s2, s3);
    let (s10, d10) = fwht_2(s4, s5);
    let (s11, d11) = fwht_2(s6, s7);
    let (s12, d12) = fwht_2(d0, d1);
    let (s13, d13) = fwht_2(d2, d3);
    let (s14, d14) = fwht_2(d4, d5);
    let (s15, d15) = fwht_2(d6, d7);

    let (s16, d16) = fwht_2(s8, s9);
    let (s17, d17) = fwht_2(s10, s11);
    let (s18, d18) = fwht_2(s12, s13);
    let (s19, d19) = fwht_2(s14, s15);
    let (s20, d20) = fwht_2(d8, d9);
    let (s21, d21) = fwht_2(d10, d11);
    let (s22, d22) = fwht_2(d12, d13);
    let (s23, d23) = fwht_2(d14, d15);

    let (s24, d24) = fwht_2(s16, s17);
    let (s25, d25) = fwht_2(s18, s19);
    let (s26, d26) = fwht_2(s20, s21);
    let (s27, d27) = fwht_2(s22, s23);
    let (s28, d28) = fwht_2(d16, d17);
    let (s29, d29) = fwht_2(d18, d19);
    let (s30, d30) = fwht_2(d20, d21);
    let (s31, d31) = fwht_2(d22, d23);

    data[t0] = s24;
    data[t1] = s25;
    data[t2] = s26;
    data[t3] = s27;
    data[t4] = s28;
    data[t5] = s29;
    data[t6] = s30;
    data[t7] = s31;
    data[t8] = d24;
    data[t9] = d25;
    data[t10] = d26;
    data[t11] = d27;
    data[t12] = d28;
    data[t13] = d29;
    data[t14] = d30;
    data[t15] = d31;
}

// ======================================================================
// FWHT - TESTS

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    // Reference implementation
    fn fwht_naive(data: &mut [GfElement; GF_ORDER]) {
        let mut dist = 1;
        let mut dist2 = 2;
        while dist2 <= data.len() {
            for r in (0..data.len()).step_by(dist2) {
                for offset in r..r + dist {
                    let sum = engine::add_mod(data[offset], data[offset + dist]);
                    let dif = engine::sub_mod(data[offset], data[offset + dist]);
                    data[offset] = sum;
                    data[offset + dist] = dif;
                }
            }

            dist = dist2;
            dist2 *= 2;
        }
    }

    #[test]
    fn test_full() {
        let mut rng = ChaCha8Rng::from_seed([0; 32]);

        let mut data1 = [(); GF_ORDER].map(|_| rng.gen());
        let mut data2 = data1;

        fwht(&mut data1, GF_ORDER);
        fwht_naive(&mut data2);

        assert_eq!(data1, data2);
    }

    #[test]
    fn test_truncated() {
        let mut rng = ChaCha8Rng::from_seed([0; 32]);
        let random: Vec<GfElement> = (0..GF_ORDER).map(|_| rng.gen()).collect();

        for nonzero_count in [
            0,
            1,
            2,
            3,
            4,
            64,
            127,
            16384 - 1,
            16384 + 1,
            GF_ORDER / 2 - 1,
            GF_ORDER / 2,
            GF_ORDER / 2 + 1,
            GF_ORDER - 4,
            GF_ORDER - 3,
            GF_ORDER - 2,
            GF_ORDER - 1,
            GF_ORDER,
        ] {
            let mut data1 = [0; GF_ORDER];

            data1[..nonzero_count].copy_from_slice(&random[..nonzero_count]);
            let mut data2 = data1;

            fwht(&mut data1, nonzero_count);
            fwht_naive(&mut data2);

            assert_eq!(data1, data2);
        }
    }

    #[test]
    fn test_8_full() {
        let mut rng = ChaCha8Rng::from_seed([0; 32]);

        let mut data1 = [(); GF_ORDER].map(|_| rng.gen());
        let mut data2 = data1;

        fwht_8_full(&mut data1);
        fwht_naive(&mut data2);

        assert_eq!(data1, data2);
    }

    #[test]
    fn test_8_truncated() {
        let mut rng = ChaCha8Rng::from_seed([0; 32]);
        let random: Vec<GfElement> = (0..GF_ORDER).map(|_| rng.gen()).collect();

        for nonzero_count in [
            0,
            1,
            2,
            3,
            4,
            64,
            127,
            16384 - 1,
            16384 + 1,
            GF_ORDER / 2 - 1,
            GF_ORDER / 2,
            GF_ORDER / 2 + 1,
            GF_ORDER - 4,
            GF_ORDER - 3,
            GF_ORDER - 2,
            GF_ORDER - 1,
            GF_ORDER,
        ] {
            let mut data1 = [0; GF_ORDER];

            data1[..nonzero_count].copy_from_slice(&random[..nonzero_count]);
            let mut data2 = data1;

            fwht_8_truncated(&mut data1, nonzero_count);
            fwht_naive(&mut data2);

            assert_eq!(data1, data2);
        }
    }

    #[test]
    fn test_16_full() {
        let mut rng = ChaCha8Rng::from_seed([0; 32]);

        let mut data1 = [(); GF_ORDER].map(|_| rng.gen());
        let mut data2 = data1;

        fwht_16_full(&mut data1);
        fwht_naive(&mut data2);

        assert_eq!(data1, data2);
    }
}
