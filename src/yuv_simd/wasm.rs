//! WASM SIMD128 YUV → RGB24 inner loops — bit-exact twin of the Q15 scalar and NEON kernels,
//! swallowing 8 pixels per stroke so browser software playback never chokes on scalar color math.

#![allow(unsafe_op_in_unsafe_fn)]

use crate::yuv::{DecodeParams, FP_HALF, FP_SHIFT, YuvMatrix, yuv_to_rgb_fp};
use core::arch::wasm32::*;

const LANES: usize = 8;

#[inline]
#[target_feature(enable = "simd128")]
unsafe fn decode_block_i32x8(
    y_lo: v128,
    y_hi: v128,
    cb_lo: v128,
    cb_hi: v128,
    cr_lo: v128,
    cr_hi: v128,
    d: &DecodeParams,
) -> (v128, v128, v128) {
    let cr_r = i32x4_splat(d.cr_r);
    let cb_b = i32x4_splat(d.cb_b);
    let cg_cr = i32x4_splat(d.cg_cr);
    let cg_cb = i32x4_splat(d.cg_cb);
    let bias = i32x4_splat(FP_HALF);
    const SH: u32 = FP_SHIFT as u32;
    let r_lo = i32x4_shr(i32x4_add(i32x4_add(y_lo, i32x4_mul(cr_r, cr_lo)), bias), SH);
    let r_hi = i32x4_shr(i32x4_add(i32x4_add(y_hi, i32x4_mul(cr_r, cr_hi)), bias), SH);
    let b_lo = i32x4_shr(i32x4_add(i32x4_add(y_lo, i32x4_mul(cb_b, cb_lo)), bias), SH);
    let b_hi = i32x4_shr(i32x4_add(i32x4_add(y_hi, i32x4_mul(cb_b, cb_hi)), bias), SH);
    let g_lo = i32x4_shr(
        i32x4_add(
            i32x4_sub(
                i32x4_sub(y_lo, i32x4_mul(cg_cr, cr_lo)),
                i32x4_mul(cg_cb, cb_lo),
            ),
            bias,
        ),
        SH,
    );
    let g_hi = i32x4_shr(
        i32x4_add(
            i32x4_sub(
                i32x4_sub(y_hi, i32x4_mul(cg_cr, cr_hi)),
                i32x4_mul(cg_cb, cb_hi),
            ),
            bias,
        ),
        SH,
    );

    let r16 = i16x8_narrow_i32x4(r_lo, r_hi);
    let g16 = i16x8_narrow_i32x4(g_lo, g_hi);
    let b16 = i16x8_narrow_i32x4(b_lo, b_hi);
    let zero = i16x8_splat(0);
    (
        u8x16_narrow_i16x8(r16, zero),
        u8x16_narrow_i16x8(g16, zero),
        u8x16_narrow_i16x8(b16, zero),
    )
}

#[inline]
#[target_feature(enable = "simd128")]
unsafe fn load8_sub_i32(src: &[u8], off: i32) -> (v128, v128) {
    let raw = u16x8_extend_low_u8x16(v128_load64_zero(src.as_ptr() as *const u64));
    let lo = u32x4_extend_low_u16x8(raw);
    let hi = u32x4_extend_high_u16x8(raw);
    let off_v = i32x4_splat(off);
    (i32x4_sub(lo, off_v), i32x4_sub(hi, off_v))
}

#[inline]
#[target_feature(enable = "simd128")]
unsafe fn load_chroma_4_broadcast(src: &[u8]) -> (v128, v128) {
    let bytes = [
        src[0], src[0], src[1], src[1], src[2], src[2], src[3], src[3],
    ];
    load8_sub_i32(&bytes, 128)
}

#[inline]
#[target_feature(enable = "simd128")]
unsafe fn store_rgb24_lane8(dst: &mut [u8], r: v128, g: v128, b: v128) {
    // Interleave low 8 bytes of r, g, b into 24 tight RGB24 bytes (16 + 8).
    let rg = i8x16_shuffle::<0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23>(r, g);
    let b0 = b;
    let out0 = i8x16_shuffle::<0, 1, 16, 2, 3, 17, 4, 5, 18, 6, 7, 19, 8, 9, 20, 10>(rg, b0);
    let out1 = i8x16_shuffle::<11, 21, 12, 13, 22, 14, 15, 23, 0, 0, 0, 0, 0, 0, 0, 0>(rg, b0);
    v128_store(dst.as_mut_ptr() as *mut v128, out0);
    v128_store64_lane::<0>(out1, dst.as_mut_ptr().add(16) as *mut u64);
}

#[target_feature(enable = "simd128")]
pub(crate) unsafe fn yuv444_to_rgb24(
    yp: &[u8],
    up: &[u8],
    vp: &[u8],
    dst: &mut [u8],
    w: usize,
    h: usize,
    matrix: YuvMatrix,
) {
    let d = matrix.decode_params();
    let y_scale = i32x4_splat(d.y_scale);
    for row in 0..h {
        let yrow = &yp[row * w..row * w + w];
        let urow = &up[row * w..row * w + w];
        let vrow = &vp[row * w..row * w + w];
        let drow = &mut dst[row * w * 3..row * w * 3 + w * 3];
        let chunks = w / LANES;
        for chunk in 0..chunks {
            let off = chunk * LANES;
            let (y0, y1) = load8_sub_i32(&yrow[off..], d.y_off);
            let (cb0, cb1) = load8_sub_i32(&urow[off..], 128);
            let (cr0, cr1) = load8_sub_i32(&vrow[off..], 128);
            let (rv, gv, bv) = decode_block_i32x8(
                i32x4_mul(y0, y_scale),
                i32x4_mul(y1, y_scale),
                cb0,
                cb1,
                cr0,
                cr1,
                &d,
            );
            store_rgb24_lane8(&mut drow[off * 3..off * 3 + LANES * 3], rv, gv, bv);
        }
        for col in (chunks * LANES)..w {
            let (r, g, b) = yuv_to_rgb_fp(yrow[col], urow[col], vrow[col], &d);
            drow[col * 3] = r;
            drow[col * 3 + 1] = g;
            drow[col * 3 + 2] = b;
        }
    }
}

#[target_feature(enable = "simd128")]
pub(crate) unsafe fn yuv422_to_rgb24(
    yp: &[u8],
    up: &[u8],
    vp: &[u8],
    dst: &mut [u8],
    w: usize,
    h: usize,
    matrix: YuvMatrix,
) {
    let cw = w.div_ceil(2);
    let d = matrix.decode_params();
    let y_scale = i32x4_splat(d.y_scale);
    for row in 0..h {
        let yrow = &yp[row * w..row * w + w];
        let urow = &up[row * cw..row * cw + cw];
        let vrow = &vp[row * cw..row * cw + cw];
        let drow = &mut dst[row * w * 3..row * w * 3 + w * 3];
        let chunks = w / LANES;
        for chunk in 0..chunks {
            let off = chunk * LANES;
            let coff = off / 2;
            let (y0, y1) = load8_sub_i32(&yrow[off..], d.y_off);
            let (cb0, cb1) = load_chroma_4_broadcast(&urow[coff..]);
            let (cr0, cr1) = load_chroma_4_broadcast(&vrow[coff..]);
            let (rv, gv, bv) = decode_block_i32x8(
                i32x4_mul(y0, y_scale),
                i32x4_mul(y1, y_scale),
                cb0,
                cb1,
                cr0,
                cr1,
                &d,
            );
            store_rgb24_lane8(&mut drow[off * 3..off * 3 + LANES * 3], rv, gv, bv);
        }
        for col in (chunks * LANES)..w {
            let cc = col / 2;
            let (r, g, b) = yuv_to_rgb_fp(yrow[col], urow[cc], vrow[cc], &d);
            drow[col * 3] = r;
            drow[col * 3 + 1] = g;
            drow[col * 3 + 2] = b;
        }
    }
}

#[target_feature(enable = "simd128")]
pub(crate) unsafe fn yuv420_to_rgb24(
    yp: &[u8],
    up: &[u8],
    vp: &[u8],
    dst: &mut [u8],
    w: usize,
    h: usize,
    matrix: YuvMatrix,
) {
    let cw = w.div_ceil(2);
    let d = matrix.decode_params();
    let y_scale = i32x4_splat(d.y_scale);
    for row in 0..h {
        let cr = row / 2;
        let yrow = &yp[row * w..row * w + w];
        let urow = &up[cr * cw..cr * cw + cw];
        let vrow = &vp[cr * cw..cr * cw + cw];
        let drow = &mut dst[row * w * 3..row * w * 3 + w * 3];
        let chunks = w / LANES;
        for chunk in 0..chunks {
            let off = chunk * LANES;
            let coff = off / 2;
            let (y0, y1) = load8_sub_i32(&yrow[off..], d.y_off);
            let (cb0, cb1) = load_chroma_4_broadcast(&urow[coff..]);
            let (cr0, cr1) = load_chroma_4_broadcast(&vrow[coff..]);
            let (rv, gv, bv) = decode_block_i32x8(
                i32x4_mul(y0, y_scale),
                i32x4_mul(y1, y_scale),
                cb0,
                cb1,
                cr0,
                cr1,
                &d,
            );
            store_rgb24_lane8(&mut drow[off * 3..off * 3 + LANES * 3], rv, gv, bv);
        }
        for col in (chunks * LANES)..w {
            let cc = col / 2;
            let (r, g, b) = yuv_to_rgb_fp(yrow[col], urow[cc], vrow[cc], &d);
            drow[col * 3] = r;
            drow[col * 3 + 1] = g;
            drow[col * 3 + 2] = b;
        }
    }
}
