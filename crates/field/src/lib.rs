use num_bigint::BigUint;
use num_integer::Integer;
use num_traits::{One, Zero};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::ops::{Add, Mul, Neg, Sub};

/// Parameters for the finite field F_p where p = e*g + 1.
/// e = 2^lambda, and G is the unique subgroup of F_p* of order e.
#[derive(Clone, Debug)]
pub struct FieldParams {
    pub p: BigUint,
    pub e: BigUint,
    pub g: BigUint,
    pub lambda: u32,
}

impl FieldParams {
    /// Create field parameters from lambda. Finds a suitable prime p = e*g + 1
    /// where e = 2^lambda and p has bit-length roughly 2*lambda.
    pub fn new(lambda: u32) -> Self {
        let e = BigUint::one() << lambda;
        // For prototype, find a prime p = e*g + 1 with small g.
        // We search for odd g such that p is prime.
        let mut g = BigUint::from(3u32);
        loop {
            let p = &e * &g + BigUint::one();
            if is_prime(&p) {
                return FieldParams { p, e, g, lambda };
            }
            g += BigUint::from(2u32);
        }
    }

    /// Create field params from an explicit prime p and lambda.
    /// Verifies that p = e*g + 1 where e = 2^lambda.
    pub fn from_prime(p: BigUint, lambda: u32) -> Option<Self> {
        let e = BigUint::one() << lambda;
        let p_minus_1 = &p - BigUint::one();
        let (g, rem) = p_minus_1.div_rem(&e);
        if !rem.is_zero() {
            return None;
        }
        Some(FieldParams { p, e, g, lambda })
    }
}

/// A field element in F_p.
#[derive(Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Fp {
    pub value: BigUint,
    #[serde(skip)]
    modulus: BigUint,
}

impl fmt::Debug for Fp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Fp({})", self.value)
    }
}

impl fmt::Display for Fp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.value)
    }
}

impl Fp {
    pub fn new(value: BigUint, modulus: &BigUint) -> Self {
        Fp {
            value: value % modulus,
            modulus: modulus.clone(),
        }
    }

    pub fn zero(modulus: &BigUint) -> Self {
        Fp {
            value: BigUint::zero(),
            modulus: modulus.clone(),
        }
    }

    pub fn one(modulus: &BigUint) -> Self {
        Fp {
            value: BigUint::one(),
            modulus: modulus.clone(),
        }
    }

    pub fn modulus(&self) -> &BigUint {
        &self.modulus
    }

    /// Set the modulus (used after deserialization).
    pub fn set_modulus(&mut self, modulus: &BigUint) {
        self.modulus = modulus.clone();
        self.value = &self.value % modulus;
    }

    pub fn is_zero(&self) -> bool {
        self.value.is_zero()
    }

    pub fn add_fp(&self, other: &Fp) -> Fp {
        debug_assert_eq!(self.modulus, other.modulus);
        let sum = &self.value + &other.value;
        Fp::new(sum, &self.modulus)
    }

    pub fn sub_fp(&self, other: &Fp) -> Fp {
        debug_assert_eq!(self.modulus, other.modulus);
        let val = if self.value >= other.value {
            &self.value - &other.value
        } else {
            &self.modulus - &other.value + &self.value
        };
        Fp::new(val, &self.modulus)
    }

    pub fn mul_fp(&self, other: &Fp) -> Fp {
        debug_assert_eq!(self.modulus, other.modulus);
        let prod = &self.value * &other.value;
        Fp::new(prod, &self.modulus)
    }

    pub fn neg_fp(&self) -> Fp {
        if self.value.is_zero() {
            self.clone()
        } else {
            Fp {
                value: &self.modulus - &self.value,
                modulus: self.modulus.clone(),
            }
        }
    }

    /// Modular exponentiation: self^exp mod p.
    pub fn pow(&self, exp: &BigUint) -> Fp {
        Fp {
            value: self.value.modpow(exp, &self.modulus),
            modulus: self.modulus.clone(),
        }
    }

    /// Modular inverse using Fermat's little theorem: a^{-1} = a^{p-2} mod p.
    /// Returns None if self is zero.
    pub fn inv(&self) -> Option<Fp> {
        if self.value.is_zero() {
            return None;
        }
        let exp = &self.modulus - BigUint::from(2u32);
        Some(self.pow(&exp))
    }

    /// Generate a random field element in [0, p).
    pub fn random(modulus: &BigUint, rng: &mut impl Rng) -> Fp {
        let bit_len = modulus.bits();
        loop {
            let bytes_needed = ((bit_len + 7) / 8) as usize;
            let mut bytes = vec![0u8; bytes_needed];
            rng.fill(&mut bytes[..]);
            // Mask the top byte to avoid too-large values
            let excess_bits = (bytes_needed * 8) as u64 - bit_len;
            if excess_bits > 0 {
                bytes[0] >>= excess_bits;
            }
            let val = BigUint::from_bytes_be(&bytes);
            if val < *modulus {
                return Fp {
                    value: val,
                    modulus: modulus.clone(),
                };
            }
        }
    }

    /// Generate a random non-zero field element.
    pub fn random_nonzero(modulus: &BigUint, rng: &mut impl Rng) -> Fp {
        loop {
            let r = Fp::random(modulus, rng);
            if !r.is_zero() {
                return r;
            }
        }
    }
}

// Operator overloads for convenience.
impl Add for &Fp {
    type Output = Fp;
    fn add(self, rhs: &Fp) -> Fp {
        self.add_fp(rhs)
    }
}

impl Sub for &Fp {
    type Output = Fp;
    fn sub(self, rhs: &Fp) -> Fp {
        self.sub_fp(rhs)
    }
}

impl Mul for &Fp {
    type Output = Fp;
    fn mul(self, rhs: &Fp) -> Fp {
        self.mul_fp(rhs)
    }
}

impl Neg for &Fp {
    type Output = Fp;
    fn neg(self) -> Fp {
        self.neg_fp()
    }
}

/// Simple Miller-Rabin primality test for BigUint.
pub fn is_prime(n: &BigUint) -> bool {
    if *n < BigUint::from(2u32) {
        return false;
    }
    if *n == BigUint::from(2u32) || *n == BigUint::from(3u32) {
        return true;
    }
    if n.is_even() {
        return false;
    }

    // Write n-1 = 2^r * d
    let n_minus_1 = n - BigUint::one();
    let mut d = n_minus_1.clone();
    let mut r = 0u32;
    while d.is_even() {
        d >>= 1;
        r += 1;
    }

    // Test with small bases
    let bases: Vec<BigUint> = vec![2u32, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        .into_iter()
        .map(BigUint::from)
        .collect();

    'outer: for a in &bases {
        if a >= n {
            continue;
        }
        let mut x = a.modpow(&d, n);
        if x == BigUint::one() || x == n_minus_1 {
            continue;
        }
        for _ in 0..r - 1 {
            x = x.modpow(&BigUint::from(2u32), n);
            if x == n_minus_1 {
                continue 'outer;
            }
        }
        return false;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_params() -> FieldParams {
        // Small field for testing: lambda=4, e=16, find p = 16*g + 1 prime
        // p = 16*2 + 1 = 33 (not prime), 16*3+1=49 (not prime), 16*5+1=81 (not), 16*7+1=113 (prime!)
        FieldParams::from_prime(BigUint::from(113u32), 4).unwrap()
    }

    #[test]
    fn test_field_params() {
        let params = test_params();
        assert_eq!(params.p, BigUint::from(113u32));
        assert_eq!(params.e, BigUint::from(16u32));
        assert_eq!(params.g, BigUint::from(7u32));
    }

    #[test]
    fn test_add_sub() {
        let p = BigUint::from(113u32);
        let a = Fp::new(BigUint::from(50u32), &p);
        let b = Fp::new(BigUint::from(80u32), &p);
        let sum = &a + &b;
        assert_eq!(sum.value, BigUint::from(17u32)); // (50+80) mod 113 = 17
        let diff = &a - &b;
        assert_eq!(diff.value, BigUint::from(83u32)); // (50-80+113) mod 113 = 83
    }

    #[test]
    fn test_mul() {
        let p = BigUint::from(113u32);
        let a = Fp::new(BigUint::from(10u32), &p);
        let b = Fp::new(BigUint::from(12u32), &p);
        let prod = &a * &b;
        assert_eq!(prod.value, BigUint::from(7u32)); // (120) mod 113 = 7
    }

    #[test]
    fn test_inv() {
        let p = BigUint::from(113u32);
        let a = Fp::new(BigUint::from(10u32), &p);
        let a_inv = a.inv().unwrap();
        let prod = &a * &a_inv;
        assert_eq!(prod.value, BigUint::one());
    }

    #[test]
    fn test_pow() {
        let p = BigUint::from(113u32);
        let a = Fp::new(BigUint::from(3u32), &p);
        let result = a.pow(&BigUint::from(4u32));
        assert_eq!(result.value, BigUint::from(81u32)); // 3^4 = 81 < 113
    }

    #[test]
    fn test_neg() {
        let p = BigUint::from(113u32);
        let a = Fp::new(BigUint::from(50u32), &p);
        let neg_a = -&a;
        let sum = &a + &neg_a;
        assert!(sum.is_zero());
    }

    #[test]
    fn test_zero_inv_returns_none() {
        let p = BigUint::from(113u32);
        let z = Fp::zero(&p);
        assert!(z.inv().is_none());
    }

    #[test]
    fn test_random_in_range() {
        let p = BigUint::from(113u32);
        let mut rng = rand::thread_rng();
        for _ in 0..100 {
            let r = Fp::random(&p, &mut rng);
            assert!(r.value < p);
        }
    }

    #[test]
    fn test_fermat_little_theorem() {
        // a^{p-1} = 1 mod p for a != 0
        let p = BigUint::from(113u32);
        let a = Fp::new(BigUint::from(42u32), &p);
        let p_minus_1 = &p - BigUint::one();
        let result = a.pow(&p_minus_1);
        assert_eq!(result.value, BigUint::one());
    }

    #[test]
    fn test_is_prime() {
        assert!(is_prime(&BigUint::from(113u32)));
        assert!(is_prime(&BigUint::from(127u32)));
        assert!(!is_prime(&BigUint::from(100u32)));
        assert!(!is_prime(&BigUint::from(1u32)));
    }

    #[test]
    fn test_field_params_new() {
        let params = FieldParams::new(4);
        // e = 16, should find a prime p = 16*g + 1
        assert_eq!(params.e, BigUint::from(16u32));
        assert!(is_prime(&params.p));
        assert_eq!(&params.e * &params.g + BigUint::one(), params.p);
    }
}
