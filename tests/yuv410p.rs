//! 4:1:0 planar (`Yuv410P`, FFmpeg's yuv410p: SVQ1 and Indeo output) at
//! any size: each finer chroma sample is the 4:1:0 sample whose 4×4 luma
//! block holds it, luma passes unchanged, and the targets the players
//! render from (NV12 on Apple, RGBA on Android) are reached.

use oxideav_core::{PixelFormat, VideoFrame, VideoPlane};
use oxideav_pixfmt::{convert, ConvertOptions, FrameInfo};

/// A `w`×`h` 4:1:0 frame with distinct luma and chroma samples.
fn frame(w: u32, h: u32) -> VideoFrame {
    let (w, h) = (w as usize, h as usize);
    let (cw, ch) = (w.div_ceil(4), h.div_ceil(4));
    let luma = (0..w * h).map(|i| (i * 7 % 251) as u8).collect();
    let u = (0..cw * ch).map(|i| (i * 13 % 241) as u8).collect();
    let v = (0..cw * ch).map(|i| (255 - i * 11 % 239) as u8).collect();
    VideoFrame {
        pts: None,
        planes: vec![
            VideoPlane { stride: w, data: luma },
            VideoPlane { stride: cw, data: u },
            VideoPlane { stride: cw, data: v },
        ],
    }
}

fn to(f: &VideoFrame, w: u32, h: u32, dst: PixelFormat) -> VideoFrame {
    convert(f, FrameInfo::new(PixelFormat::Yuv410P, w, h), dst, &ConvertOptions::default()).unwrap()
}

/// Sorenson Video 1's 293x178 (FATE `svq1/ct_ending_cut.mov`), a tiny
/// odd size and a block-aligned one.
const SIZES: [(u32, u32); 3] = [(293, 178), (7, 5), (160, 120)];

#[test]
fn to_420_422_444_duplicates_each_chroma_sample_over_its_block() {
    for (w, h) in SIZES {
        let src = frame(w, h);
        let scw = (w as usize).div_ceil(4);
        for (dst, dw, dh) in [(PixelFormat::Yuv420P, 2, 2), (PixelFormat::Yuv422P, 2, 1), (PixelFormat::Yuv444P, 1, 1)] {
            let out = to(&src, w, h, dst);
            assert_eq!(out.planes[0].data, src.planes[0].data, "{w}x{h} {dst:?}: luma");
            let (dcw, dch) = ((w as usize).div_ceil(dw), (h as usize).div_ceil(dh));
            for p in 1..3 {
                let plane = &out.planes[p];
                assert_eq!((plane.stride, plane.data.len()), (dcw, dcw * dch), "{w}x{h} {dst:?}: plane {p} size");
                for row in 0..dch {
                    for col in 0..dcw {
                        let want = src.planes[p].data[row * dh / 4 * scw + col * dw / 4];
                        assert_eq!(plane.data[row * dcw + col], want, "{w}x{h} {dst:?}: plane {p} ({col}, {row})");
                    }
                }
            }
        }
    }
}

/// NV12 interleaves the 4:2:0 chroma: each pair is the U and V sample of
/// the 4:1:0 block holding it.
#[test]
fn to_nv12_for_the_apple_renderer() {
    for (w, h) in SIZES {
        let src = frame(w, h);
        let out = to(&src, w, h, PixelFormat::Nv12);
        assert_eq!(out.planes[0].data, src.planes[0].data, "{w}x{h}: luma");
        let scw = (w as usize).div_ceil(4);
        let (dcw, dch) = ((w as usize).div_ceil(2), (h as usize).div_ceil(2));
        let uv = &out.planes[1];
        for row in 0..dch {
            for col in 0..dcw {
                let at = row / 2 * scw + col / 2;
                let got = (uv.data[row * uv.stride + 2 * col], uv.data[row * uv.stride + 2 * col + 1]);
                assert_eq!(got, (src.planes[1].data[at], src.planes[2].data[at]), "{w}x{h}: chroma ({col}, {row})");
            }
        }
    }
}

#[test]
fn to_rgba_and_gray() {
    for (w, h) in SIZES {
        let src = frame(w, h);
        let rgba = to(&src, w, h, PixelFormat::Rgba);
        assert_eq!(rgba.planes[0].data.len(), (w * h * 4) as usize, "{w}x{h}: RGBA");
        let gray = to(&src, w, h, PixelFormat::Gray8);
        assert_eq!(gray.planes[0].data.len(), (w * h) as usize, "{w}x{h}: gray");
    }
}
