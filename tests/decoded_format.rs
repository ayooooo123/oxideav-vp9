//! The registry decoder reports the size and pixel layout of the frame it
//! last returned (oxideav-core `Decoder::output_video_dimensions` /
//! `output_pixel_format`): odd sizes, 10-bit and 4:4:4. Frames are
//! decoded as packets arrive, so frames of an earlier size still queued
//! behind a new keyframe report their own size.
//!
//! `decoded_format/*.ivf` are two-frame libvpx-vp9 encodes of FFmpeg
//! `testsrc` (profiles 0, 2 and 1).

use oxideav_core::{Decoder, Error, Frame, Packet, PixelFormat, TimeBase, VideoFrame};

/// A frame's width, height and pixel format.
type Layout = (u32, u32, PixelFormat);

/// The frames of every stream, in order, each with its layout.
fn frames() -> Vec<(Vec<u8>, Layout)> {
    use PixelFormat::*;
    let streams: [(&[u8], Layout); 3] = [
        (
            include_bytes!("decoded_format/a_420_33x17.ivf"),
            (33, 17, Yuv420P),
        ),
        (
            include_bytes!("decoded_format/b_420p10_34x18.ivf"),
            (34, 18, Yuv420P10Le),
        ),
        (
            include_bytes!("decoded_format/c_444_35x19.ivf"),
            (35, 19, Yuv444P),
        ),
    ];
    let mut out = Vec::new();
    for (file, layout) in streams {
        // IVF: a file header of the length at bytes 6..8, then a 12-byte
        // header (size, pts) before each frame.
        let mut at = usize::from(u16::from_le_bytes([file[6], file[7]]));
        while at + 12 <= file.len() {
            let len = u32::from_le_bytes(file[at..at + 4].try_into().unwrap()) as usize;
            out.push((file[at + 12..at + 12 + len].to_vec(), layout));
            at += 12 + len;
        }
    }
    assert_eq!(out.len(), 6);
    out
}

fn report(dec: &dyn Decoder) -> Option<Layout> {
    let (w, h) = dec.output_video_dimensions()?;
    Some((w, h, dec.output_pixel_format()?))
}

fn assert_geometry(frame: &VideoFrame, (w, h, format): Layout, at: usize) {
    let bytes = if format == PixelFormat::Yuv420P10Le {
        2
    } else {
        1
    };
    let planes = frame.image_planes();
    assert_eq!(planes.len(), 3, "frame {at}: planes");
    for (i, plane) in planes.iter().enumerate() {
        let (pw, ph) = format.plane_dimensions(i, w, h).unwrap();
        assert_eq!(
            plane.stride,
            pw as usize * bytes,
            "frame {at}: plane {i} stride"
        );
        assert_eq!(
            plane.data.len(),
            plane.stride * ph as usize,
            "frame {at}: plane {i} rows"
        );
    }
}

/// Receives every available frame, checking each right after it is
/// returned.
fn receive_all(dec: &mut dyn Decoder, expected: &[Layout], seen: &mut usize) {
    loop {
        match dec.receive_frame() {
            Ok(Frame::Video(frame)) => {
                assert_eq!(
                    report(dec),
                    Some(expected[*seen]),
                    "report after frame {seen}"
                );
                assert_geometry(&frame, expected[*seen], *seen);
                *seen += 1;
            }
            Ok(_) => panic!("non-video frame"),
            Err(Error::NeedMore | Error::Eof) => return,
            Err(e) => panic!("receive: {e}"),
        }
    }
}

/// Every packet sent before any frame is received, including threaded drain.
#[test]
fn each_frame_reports_its_own_size_and_layout() {
    let (packets, expected): (Vec<_>, Vec<_>) = frames().into_iter().unzip();
    for threads in [1, 8] {
        let mut dec = oxideav_vp9::Vp9Decoder::new();
        dec.set_execution_context(&oxideav_core::ExecutionContext::with_threads(threads));
        assert_eq!(report(&dec), None, "fresh decoder");
        for data in &packets {
            dec.send_packet(&Packet::new(0, TimeBase::new(1, 25), data.clone()))
                .expect("send");
        }
        if threads == 1 {
            assert_eq!(report(&dec), Some(expected[0]), "before the first frame");
        }
        dec.flush().expect("flush");
        let mut seen = 0;
        receive_all(&mut dec, &expected, &mut seen);
        assert_eq!(seen, expected.len(), "frames at {threads} threads");
        assert_eq!(report(&dec), expected.last().copied(), "after drain");
        dec.reset().expect("reset");
        assert_eq!(report(&dec), None, "after reset");
    }
}

/// One packet at a time, receiving after each, with delayed threaded output.
#[test]
fn reports_change_with_the_frame_that_carries_the_change() {
    let (packets, expected): (Vec<_>, Vec<_>) = frames().into_iter().unzip();
    for threads in [1, 8] {
        let mut dec = oxideav_vp9::Vp9Decoder::new();
        dec.set_execution_context(&oxideav_core::ExecutionContext::with_threads(threads));
        let mut seen = 0;
        for data in &packets {
            dec.send_packet(&Packet::new(0, TimeBase::new(1, 25), data.clone()))
                .expect("send");
            receive_all(&mut dec, &expected, &mut seen);
        }
        dec.flush().expect("flush");
        receive_all(&mut dec, &expected, &mut seen);
        assert_eq!(seen, expected.len(), "frames at {threads} threads");
    }
}
