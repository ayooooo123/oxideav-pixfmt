use oxideav_core::{PixelFormat, VideoFrame, VideoPlane};
use oxideav_pixfmt::{convert, ConvertOptions, FrameInfo};

fn frame(bits: u32, planes: usize, masked: bool) -> VideoFrame {
    let mask = ((1u32 << bits) - 1) as u16;
    let data: Vec<u8> = (0u32..=65535).flat_map(|value| {
        let word = value as u16;
        (if masked { word & mask } else { word }).to_le_bytes()
    }).collect();
    VideoFrame { pts: None, planes: (0..planes).map(|_| VideoPlane { stride: 512, data: data.clone() }).collect() }
}

#[test]
fn existing_deep_routes_ignore_overrange_padding_bits() {
    use PixelFormat::*;
    let destinations = [Gray8, Gray10Le, Gray12Le, Gray16Le, Rgb24, Rgb48Le, Yuv420P, Yuv420P10Le, Yuv420P12Le, Yuv444P16Le, Gbrp10Le, Gbrp12Le, Gbrp16Le];
    for (source, bits, planes) in [(Gray10Le, 10, 1), (Gray12Le, 12, 1), (Gbrp10Le, 10, 3), (Gbrp12Le, 12, 3), (Yuv444P10Le, 10, 3), (Yuv444P12Le, 12, 3)] {
        let info = FrameInfo::new(source, 256, 256);
        let dirty = frame(bits, planes, false);
        let clean = frame(bits, planes, true);
        for destination in destinations {
            if source == destination { continue; }
            let expected = convert(&clean, info, destination, &ConvertOptions::default()).unwrap();
            let actual = convert(&dirty, info, destination, &ConvertOptions::default()).unwrap();
            let expected: Vec<_> = expected.planes.into_iter().map(|plane| (plane.stride, plane.data)).collect();
            let actual: Vec<_> = actual.planes.into_iter().map(|plane| (plane.stride, plane.data)).collect();
            assert_eq!(actual, expected, "{source:?} -> {destination:?}");
        }
    }
}
