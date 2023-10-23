//! Lookup-tables used by [`Engine`]:s.
//!
//! All tables are global and each is initialized at most once.
//!
//! # Tables
//!
//! | Table        | Size    | Used in encoding | Used in decoding | By engines         |
//! | ------------ | ------- | ---------------- | ---------------- | ------------------ |
//! | [`Exp`]      | 128 kiB | yes              | yes              | all                |
//! | [`Log`]      | 128 kiB | yes              | yes              | all                |
//! | [`LogWalsh`] | 128 kiB | -                | yes              | all                |
//! | [`Mul16`]    | 8 MiB   | yes              | yes              | [`NoSimd`]         |
//! | [`Mul128`]   | 8 MiB   | yes              | yes              | [`Avx2`] [`Ssse3`] |
//! | [`Skew`]     | 128 kiB | yes              | yes              | all                |
//!
//! [`NoSimd`]: crate::engine::NoSimd
//! [`Avx2`]: crate::engine::Avx2
//! [`Ssse3`]: crate::engine::Ssse3
//!

use once_cell::sync::OnceCell;

use crate::engine::{
    self, Engine, GfElement, CANTOR_BASIS, GF_BITS, GF_MODULUS, GF_ORDER, GF_POLYNOMIAL,
};

// ======================================================================
// TYPE ALIASES - PUBLIC

/// Used by [`Naive`] engine for multiplications
/// and by all [`Engine`]:s to initialize other tables.
///
/// [`Naive`]: crate::engine::Naive
pub type Exp = [GfElement; GF_ORDER];

/// Used by [`Naive`] engine for multiplications
/// and by all [`Engine`]:s to initialize other tables.
///
/// [`Naive`]: crate::engine::Naive
pub type Log = [GfElement; GF_ORDER];

/// Used by all [`Engine`]:s in [`Engine::eval_poly`].
pub type LogWalsh = [GfElement; GF_ORDER];

/// Used by [`NoSimd`] engine for multiplications.
///
/// [`NoSimd`]: crate::engine::NoSimd
pub type Mul16 = [[[GfElement; 16]; 4]; GF_ORDER];

/// Used by all [`Engine`]:s for FFT and IFFT.
pub type Skew = [GfElement; GF_MODULUS as usize];

/// Elements of the Mul128 table
#[derive(Clone, Debug)]
pub struct Multiply128lutT {
    /// Lower half of GfElements
    pub lo: [u128; 4],
    /// Upper half of GfElements
    pub hi: [u128; 4],
}

/// Used by [`Avx2`] and [`Ssse3`] engine for multiplications.
///
/// [`Avx2`]: crate::engine::Avx2
pub type Mul128 = [Multiply128lutT; GF_ORDER];

// ======================================================================
// ExpLog - PRIVATE

struct ExpLog {
    exp: Box<Exp>,
    log: Box<Log>,
}

// ======================================================================
// STATIC - PRIVATE

static EXP_LOG: OnceCell<ExpLog> = OnceCell::new();
static LOG_WALSH: OnceCell<Box<LogWalsh>> = OnceCell::new();
static MUL16: OnceCell<Box<Mul16>> = OnceCell::new();
static MUL128: OnceCell<Box<Mul128>> = OnceCell::new();
static SKEW: OnceCell<Box<Skew>> = OnceCell::new();

// ======================================================================
// FUNCTIONS - PUBLIC - math

/// Calculates `x * log_m` using [`Exp`] and [`Log`] tables.
#[inline(always)]
pub fn mul(x: GfElement, log_m: GfElement, exp: &Exp, log: &Log) -> GfElement {
    if x == 0 {
        0
    } else {
        exp[engine::add_mod(log[x as usize], log_m) as usize]
    }
}

// ======================================================================
// FUNCTIONS - PUBLIC - initialize tables

/// Initializes and returns [`Exp`] and [`Log`] tables.
#[allow(clippy::needless_range_loop)]
pub fn initialize_exp_log() -> (&'static Exp, &'static Log) {
    let exp_log = EXP_LOG.get_or_init(|| {
        let mut exp = Box::new([0; GF_ORDER]);
        let mut log = Box::new([0; GF_ORDER]);

        // GENERATE LFSR TABLE

        let mut state = 1;
        for i in 0..GF_MODULUS {
            exp[state] = i;
            state <<= 1;
            if state >= GF_ORDER {
                state ^= GF_POLYNOMIAL;
            }
        }
        exp[0] = GF_MODULUS;

        // CONVERT TO CANTOR BASIS

        log[0] = 0;
        for i in 0..GF_BITS {
            let width = 1usize << i;
            for j in 0..width {
                log[j + width] = log[j] ^ CANTOR_BASIS[i];
            }
        }

        for i in 0..GF_ORDER {
            log[i] = exp[log[i] as usize];
        }

        for i in 0..GF_ORDER {
            exp[log[i] as usize] = i as GfElement;
        }

        exp[GF_MODULUS as usize] = exp[0];

        ExpLog { exp, log }
    });

    (&exp_log.exp, &exp_log.log)
}

/// Initializes and returns [`LogWalsh`] table.
pub fn initialize_log_walsh<E: Engine>() -> &'static LogWalsh {
    LOG_WALSH.get_or_init(|| {
        let (_, log) = initialize_exp_log();

        let mut log_walsh: Box<LogWalsh> = Box::new([0; GF_ORDER]);

        log_walsh.copy_from_slice(log.as_ref());
        log_walsh[0] = 0;
        E::fwht(log_walsh.as_mut(), GF_ORDER);

        log_walsh
    })
}

/// Initializes and returns [`Mul16`] table.
pub fn initialize_mul16() -> &'static Mul16 {
    MUL16.get_or_init(|| {
        let (exp, log) = initialize_exp_log();

        let mut mul16 = vec![[[0; 16]; 4]; GF_ORDER];

        for log_m in 0..=GF_MODULUS {
            let lut = &mut mul16[log_m as usize];
            for i in 0..16 {
                lut[0][i] = mul(i as GfElement, log_m, exp, log);
                lut[1][i] = mul((i << 4) as GfElement, log_m, exp, log);
                lut[2][i] = mul((i << 8) as GfElement, log_m, exp, log);
                lut[3][i] = mul((i << 12) as GfElement, log_m, exp, log);
            }
        }

        mul16.into_boxed_slice().try_into().unwrap()
    })
}

/// Initializes and returns [`Mul128`] table.
pub fn initialize_mul128() -> &'static Mul128 {
    // Based on:
    // https://github.com/catid/leopard/blob/22ddc7804998d31c8f1a2617ee720e063b1fa6cd/LeopardFF16.cpp#L375
    MUL128.get_or_init(|| {
        let (exp, log) = initialize_exp_log();

        let mut mul128 = vec![
            Multiply128lutT {
                lo: [0; 4],
                hi: [0; 4],
            };
            GF_ORDER
        ];

        for log_m in 0..=GF_MODULUS {
            for i in 0..=3 {
                let mut prod_lo = [0u8; 16];
                let mut prod_hi = [0u8; 16];
                for x in 0..16 {
                    let prod = mul((x << (i * 4)) as GfElement, log_m, exp, log);
                    prod_lo[x] = prod as u8;
                    prod_hi[x] = (prod >> 8) as u8;
                }
                mul128[log_m as usize].lo[i] = bytemuck::cast(prod_lo);
                mul128[log_m as usize].hi[i] = bytemuck::cast(prod_hi);
            }
        }

        mul128.into_boxed_slice().try_into().unwrap()
    })
}

/// Initializes and returns [`Skew`] table.
#[allow(clippy::needless_range_loop)]
pub fn initialize_skew() -> &'static Skew {
    SKEW.get_or_init(|| {
        let (exp, log) = initialize_exp_log();

        let mut skew = Box::new([0; GF_MODULUS as usize]);

        let mut temp = [0; GF_BITS - 1];

        for i in 1..GF_BITS {
            temp[i - 1] = 1 << i;
        }

        for m in 0..GF_BITS - 1 {
            let step: usize = 1 << (m + 1);

            skew[(1 << m) - 1] = 0;

            for i in m..GF_BITS - 1 {
                let s: usize = 1 << (i + 1);
                let mut j = (1 << m) - 1;
                while j < s {
                    skew[j + s] = skew[j] ^ temp[i];
                    j += step;
                }
            }

            temp[m] =
                GF_MODULUS - log[mul(temp[m], log[(temp[m] ^ 1) as usize], exp, log) as usize];

            for i in m + 1..GF_BITS - 1 {
                let sum = engine::add_mod(log[(temp[i] ^ 1) as usize], temp[m]);
                temp[i] = mul(temp[i], sum, exp, log);
            }
        }

        for i in 0..GF_MODULUS as usize {
            skew[i] = log[skew[i] as usize];
        }

        skew
    })
}
