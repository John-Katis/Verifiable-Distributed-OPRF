#include "../../arith.h"


// Reduction modulo p
// a in [0, R) -> a in [0, p-1]
// Unlike p256/p512 (whose primes are close enough to R = 2^NBITS_FIELD to
// need a shape-specific pre-fold before the final conditional subtraction),
// our p = 2^384 - 573*2^128 + 1 satisfies R < 2p directly (R - p =
// 573*2^128 - 1, tiny relative to p), so any a in [0, R) is already < 2p:
// a single conditional subtract-p suffices, no pre-correction needed.
void f_red(f_elm_t a)
{
    digit_t mask, borrow = 0, carry = 0;

    for (int i = 0; i < WORDS_FIELD; i++)
        SUBC(borrow, a[i], p[i], a[i]);

    mask = 0 - borrow;

    for (int i = 0; i < WORDS_FIELD; i++)
        ADDC(carry, a[i], p[i] & mask, a[i]);

}


// Addition of two field elements
void f_add(const f_elm_t a, const f_elm_t b, f_elm_t c)
{
    digit_t mask, carry = 0;

    for (int i = 0; i < WORDS_FIELD; i++)
        ADDC(carry, a[i], b[i], c[i]);

    mask = 0 - carry;
    carry = 0;
    for (int i = 0; i < WORDS_FIELD; i++)
        ADDC(carry, c[i], Mont_one[i] & mask, c[i]);

    f_red(c);
}

// Subtraction of two field elements
void f_sub(const f_elm_t a, const f_elm_t b, f_elm_t c)
{
    digit_t mask, borrow = 0, carry = 0;

    for (int i = 0; i < WORDS_FIELD; i++)
        SUBC(borrow, a[i], b[i], c[i]);

    mask = 0 - borrow;

    for (int i = 0; i < WORDS_FIELD; i++)
        SUBC(carry, c[i], Mont_one[i] & mask, c[i])

    f_red(c);
}

// Negation of a field element
void f_neg(const f_elm_t a, f_elm_t b)
{
    digit_t borrow = 0;

    for (int i = 0; i < WORDS_FIELD; i++)
        SUBC(borrow, p[i], a[i], b[i]);

    f_red(b);
}

// Multiplication of two multiprecision words (without reduction)
void mp_mul(const digit_t *a, const digit_t *b, digit_t *c)
{ // Schoolbook multiplication
    digit_t carry, th, tl, t = 0, u = 0, v = 0;

    for (int i = 0; i < WORDS_FIELD; i++)
    {
        for (int j = 0; j <= i; j++)
        {
            carry = 0;
            MUL(a[j], b[i - j], th, tl);
            ADDC(carry, tl, v, v);
            ADDC(carry, th, u, u);
            t += carry;
        }
        c[i] = v;
        v = u;
        u = t;
        t = 0;
    }

    for (int i = WORDS_FIELD; i < 2 * WORDS_FIELD - 1; i++)
    {
        for (int j = i - WORDS_FIELD + 1; j < WORDS_FIELD; j++)
        {
            carry = 0;
            MUL(a[j], b[i - j], th, tl);
            ADDC(carry, tl, v, v);
            ADDC(carry, th, u, u);
            t += carry;
        }
        c[i] = v;
        v = u;
        u = t;
        t = 0;
    }
    c[2 * WORDS_FIELD - 1] = v;
}

// Montgomery form reduction after multiplication
void mont_redc(const digit_t *a, digit_t *c)
{
    // c = a*R^-1 mod p, where R = 2^384.
    // If a < 2^384*p, the output c is in the range [0, p).
    // a is assumed to be in Montgomery representation.
    digit_t mask, carry = 0;
    digit_t t0[2 * WORDS_FIELD], t1[WORDS_FIELD];

    mp_mul(a, ip, t0);
    f_copy(t0, t1);
    mp_mul(t1, p, t0);

    for (int i = 0; i < 2 * WORDS_FIELD; i++)
        SUBC(carry, a[i], t0[i], t0[i]);

    mask = 0 - carry;
    carry = 0;
    for (int i = 0; i < WORDS_FIELD; i++)
        ADDC(carry, t0[WORDS_FIELD + i], p[i] & mask, c[i]);

}

// Multiplication of field elements
void f_mul(const f_elm_t a, const f_elm_t b, f_elm_t c)
{
    digit_t t0[2 * WORDS_FIELD] = {0};

    mp_mul(a, b, t0);
    mont_redc(t0, c);
}


// Convert a number from value to Montgomery form  (a -> aR)
void to_mont(const digit_t *a, f_elm_t b)
{
    f_mul(a, R2, b);
}


// Convert a number from Montgomery form into value (aR -> a)
void from_mont(const f_elm_t a, digit_t *b)
{
    digit_t t0[2 * WORDS_FIELD] = {0};
    f_copy(a, t0);
    mont_redc(t0, b);
}


// ---------------------------------------------------------------------------
// f_inv / f_leg below use a plain, generic left-to-right square-and-multiply
// exponentiation instead of the hand-derived fixed addition chains p256/p512
// use for *their own* primes' specific bit patterns (those chains are
// tailored to each prime's exact 1-bits layout in p-2 / (p-1)/2 and are not
// portable to a different prime). This is the standard, always-correct
// textbook algorithm — same complexity class, just not hand-optimized —
// which is the right trade-off here: this is a research comparison
// baseline, not a performance-critical production path.
// ---------------------------------------------------------------------------

// b = a^e mod p (Montgomery domain in, Montgomery domain out), where e is a
// WORDS_FIELD-word plain (non-Montgomery) exponent, MSB-first.
static void f_pow(const f_elm_t a, const digit_t *e, f_elm_t b)
{
    f_elm_t acc;
    f_copy(Mont_one, acc); // acc = Mont(1)

    for (int w = WORDS_FIELD - 1; w >= 0; w--)
    {
        for (int bit = 63; bit >= 0; bit--)
        {
            f_mul(acc, acc, acc);
            if ((e[w] >> bit) & 1ULL)
                f_mul(acc, a, acc);
        }
    }
    f_copy(acc, b);
}

#if (PRIMES == ORIGINAL)

// Multiplicative inverse of a field element: b = a^(p-2) mod p (Fermat).
void f_inv(const f_elm_t a, f_elm_t b)
{
    digit_t e[WORDS_FIELD];
    digit_t borrow = 0;
    digit_t one[WORDS_FIELD] = {0};
    one[0] = 1;

    // e = pm1 - 1 = (p - 1) - 1 = p - 2
    for (int i = 0; i < WORDS_FIELD; i++)
        SUBC(borrow, pm1[i], one[i], e[i]);

    f_pow(a, e, b);
}

// Legendre symbol of a field element: 0 if a is a QR (a^((p-1)/2) == 1),
// 1 otherwise (a^((p-1)/2) == -1). a == 0 is not a meaningful input for this
// protocol (never evaluated on 0) and is not specially handled.
void f_leg(const f_elm_t a, unsigned char *b)
{
    digit_t e[WORDS_FIELD];

    // e = pm1 >> 1 = (p - 1) / 2
    for (int i = 0; i < WORDS_FIELD - 1; i++)
        e[i] = (pm1[i] >> 1) | (pm1[i + 1] << 63);
    e[WORDS_FIELD - 1] = pm1[WORDS_FIELD - 1] >> 1;

    f_elm_t r;
    f_pow(a, e, r);

    // r is Mont(1) for a QR, Mont(-1) = Mont(p-1) otherwise.
    *b = f_neq(r, Mont_one) ? 1 : 0;
}

// Not used by the dOPRF protocol (only f_inv/f_leg are, see dOPRF.c) and not
// implemented for this field: our prime is p ≡ 1 (mod 4), which needs a full
// Tonelli–Shanks (not the direct a^((p+1)/4) shortcut p512 uses for its
// p ≡ 3 (mod 4) prime) — real work with no caller to justify it here.
void f_sqrt(const f_elm_t a, f_elm_t b)
{
    (void)a;
    (void)b;
}

#endif
