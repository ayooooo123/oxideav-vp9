//! Hardware SIMD (AArch64 NEON & WebAssembly SIMD128) kernels for VP9's two
//! hottest stages: §8.8.5 superblock loop filtering (`67.7%` of decode time)
//! and §8.5.2.4 8-tap sub-pixel inter prediction (`14.8%` of decode time).

#![allow(clippy::too_many_arguments, clippy::needless_range_loop)]

#[cfg(any(
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
mod imp {
    use crate::adaptive_filter_strength::FilterStrength;
    use crate::filter_size::{TX_4X4, TX_16X16};

    #[cfg(target_arch = "aarch64")]
    mod vec16 {
        use core::arch::aarch64::*;

        #[derive(Clone, Copy)]
        pub struct V16(pub int16x8_t);

        impl V16 {
            #[inline(always)]
            pub fn splat(v: i16) -> Self {
                Self(unsafe { vdupq_n_s16(v) })
            }
            #[inline(always)]
            pub unsafe fn load8(p: *const u16) -> Self {
                Self(unsafe { vreinterpretq_s16_u16(vld1q_u16(p)) })
            }
            #[inline(always)]
            pub unsafe fn load4(p: *const u16) -> Self {
                Self(unsafe { vreinterpretq_s16_u16(vcombine_u16(vld1_u16(p), vdup_n_u16(0))) })
            }
            #[inline(always)]
            pub unsafe fn store8(self, p: *mut u16) {
                unsafe { vst1q_u16(p, vreinterpretq_u16_s16(self.0)) }
            }
            #[inline(always)]
            pub unsafe fn store4(self, p: *mut u16) {
                unsafe { vst1_u16(p, vget_low_u16(vreinterpretq_u16_s16(self.0))) }
            }
            #[inline(always)]
            pub fn lane_mask4() -> Self {
                Self(unsafe {
                    vreinterpretq_s16_u16(vcombine_u16(vdup_n_u16(0xffff), vdup_n_u16(0)))
                })
            }
            #[inline(always)]
            pub fn add(self, o: Self) -> Self {
                Self(unsafe { vaddq_s16(self.0, o.0) })
            }
            #[inline(always)]
            pub fn sub(self, o: Self) -> Self {
                Self(unsafe { vsubq_s16(self.0, o.0) })
            }
            #[inline(always)]
            pub fn mul3(self) -> Self {
                Self(unsafe { vmulq_n_s16(self.0, 3) })
            }
            #[inline(always)]
            pub fn abs_diff(self, o: Self) -> Self {
                Self(unsafe { vabdq_s16(self.0, o.0) })
            }
            #[inline(always)]
            pub fn shl1(self) -> Self {
                Self(unsafe { vshlq_n_s16::<1>(self.0) })
            }
            #[inline(always)]
            pub fn shr1(self) -> Self {
                Self(unsafe { vshrq_n_s16::<1>(self.0) })
            }
            #[inline(always)]
            pub fn shr3(self) -> Self {
                Self(unsafe { vshrq_n_s16::<3>(self.0) })
            }
            #[inline(always)]
            pub fn shr4(self) -> Self {
                Self(unsafe { vshrq_n_s16::<4>(self.0) })
            }
            #[inline(always)]
            pub fn clamp(self, lo: Self, hi: Self) -> Self {
                Self(unsafe { vminq_s16(vmaxq_s16(self.0, lo.0), hi.0) })
            }
            #[inline(always)]
            pub fn le(self, o: Self) -> Self {
                Self(unsafe { vreinterpretq_s16_u16(vcleq_s16(self.0, o.0)) })
            }
            #[inline(always)]
            pub fn gt(self, o: Self) -> Self {
                Self(unsafe { vreinterpretq_s16_u16(vcgtq_s16(self.0, o.0)) })
            }
            #[inline(always)]
            pub fn and(self, o: Self) -> Self {
                Self(unsafe { vandq_s16(self.0, o.0) })
            }
            #[inline(always)]
            pub fn or(self, o: Self) -> Self {
                Self(unsafe { vorrq_s16(self.0, o.0) })
            }
            #[inline(always)]
            pub fn andnot(self, excl: Self) -> Self {
                Self(unsafe { vbicq_s16(self.0, excl.0) })
            }
            #[inline(always)]
            pub fn select(mask: Self, new: Self, old: Self) -> Self {
                Self(unsafe { vbslq_s16(vreinterpretq_u16_s16(mask.0), new.0, old.0) })
            }
            #[inline(always)]
            pub fn any(self) -> bool {
                unsafe { vmaxvq_u16(vreinterpretq_u16_s16(self.0)) != 0 }
            }
            #[inline(always)]
            pub fn transpose8(r: [Self; 8]) -> [Self; 8] {
                unsafe {
                    let a0 = vtrn1q_s16(r[0].0, r[1].0);
                    let a1 = vtrn2q_s16(r[0].0, r[1].0);
                    let a2 = vtrn1q_s16(r[2].0, r[3].0);
                    let a3 = vtrn2q_s16(r[2].0, r[3].0);
                    let a4 = vtrn1q_s16(r[4].0, r[5].0);
                    let a5 = vtrn2q_s16(r[4].0, r[5].0);
                    let a6 = vtrn1q_s16(r[6].0, r[7].0);
                    let a7 = vtrn2q_s16(r[6].0, r[7].0);

                    let b0 = vreinterpretq_s64_s32(vtrn1q_s32(
                        vreinterpretq_s32_s16(a0),
                        vreinterpretq_s32_s16(a2),
                    ));
                    let b2 = vreinterpretq_s64_s32(vtrn2q_s32(
                        vreinterpretq_s32_s16(a0),
                        vreinterpretq_s32_s16(a2),
                    ));
                    let b1 = vreinterpretq_s64_s32(vtrn1q_s32(
                        vreinterpretq_s32_s16(a1),
                        vreinterpretq_s32_s16(a3),
                    ));
                    let b3 = vreinterpretq_s64_s32(vtrn2q_s32(
                        vreinterpretq_s32_s16(a1),
                        vreinterpretq_s32_s16(a3),
                    ));
                    let b4 = vreinterpretq_s64_s32(vtrn1q_s32(
                        vreinterpretq_s32_s16(a4),
                        vreinterpretq_s32_s16(a6),
                    ));
                    let b6 = vreinterpretq_s64_s32(vtrn2q_s32(
                        vreinterpretq_s32_s16(a4),
                        vreinterpretq_s32_s16(a6),
                    ));
                    let b5 = vreinterpretq_s64_s32(vtrn1q_s32(
                        vreinterpretq_s32_s16(a5),
                        vreinterpretq_s32_s16(a7),
                    ));
                    let b7 = vreinterpretq_s64_s32(vtrn2q_s32(
                        vreinterpretq_s32_s16(a5),
                        vreinterpretq_s32_s16(a7),
                    ));

                    [
                        Self(vreinterpretq_s16_s64(vtrn1q_s64(b0, b4))),
                        Self(vreinterpretq_s16_s64(vtrn1q_s64(b1, b5))),
                        Self(vreinterpretq_s16_s64(vtrn1q_s64(b2, b6))),
                        Self(vreinterpretq_s16_s64(vtrn1q_s64(b3, b7))),
                        Self(vreinterpretq_s16_s64(vtrn2q_s64(b0, b4))),
                        Self(vreinterpretq_s16_s64(vtrn2q_s64(b1, b5))),
                        Self(vreinterpretq_s16_s64(vtrn2q_s64(b2, b6))),
                        Self(vreinterpretq_s16_s64(vtrn2q_s64(b3, b7))),
                    ]
                }
            }
        }
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    mod vec16 {
        use core::arch::wasm32::*;

        #[derive(Clone, Copy)]
        pub struct V16(pub v128);

        impl V16 {
            #[inline(always)]
            pub fn splat(v: i16) -> Self {
                Self(i16x8_splat(v))
            }
            #[inline(always)]
            pub unsafe fn load8(p: *const u16) -> Self {
                Self(unsafe { v128_load(p.cast()) })
            }
            #[inline(always)]
            pub unsafe fn load4(p: *const u16) -> Self {
                Self(unsafe { v128_load64_zero(p.cast()) })
            }
            #[inline(always)]
            pub unsafe fn store8(self, p: *mut u16) {
                unsafe { v128_store(p.cast(), self.0) }
            }
            #[inline(always)]
            pub unsafe fn store4(self, p: *mut u16) {
                unsafe { v128_store64_lane::<0>(self.0, p.cast()) }
            }
            #[inline(always)]
            pub fn lane_mask4() -> Self {
                Self(u64x2(!0u64, 0))
            }
            #[inline(always)]
            pub fn add(self, o: Self) -> Self {
                Self(i16x8_add(self.0, o.0))
            }
            #[inline(always)]
            pub fn sub(self, o: Self) -> Self {
                Self(i16x8_sub(self.0, o.0))
            }
            #[inline(always)]
            pub fn mul3(self) -> Self {
                Self(i16x8_mul(self.0, i16x8_splat(3)))
            }
            #[inline(always)]
            pub fn abs_diff(self, o: Self) -> Self {
                Self(i16x8_abs(i16x8_sub(self.0, o.0)))
            }
            #[inline(always)]
            pub fn shl1(self) -> Self {
                Self(i16x8_shl(self.0, 1))
            }
            #[inline(always)]
            pub fn shr1(self) -> Self {
                Self(i16x8_shr(self.0, 1))
            }
            #[inline(always)]
            pub fn shr3(self) -> Self {
                Self(i16x8_shr(self.0, 3))
            }
            #[inline(always)]
            pub fn shr4(self) -> Self {
                Self(i16x8_shr(self.0, 4))
            }
            #[inline(always)]
            pub fn clamp(self, lo: Self, hi: Self) -> Self {
                Self(i16x8_min(i16x8_max(self.0, lo.0), hi.0))
            }
            #[inline(always)]
            pub fn le(self, o: Self) -> Self {
                Self(i16x8_le(self.0, o.0))
            }
            #[inline(always)]
            pub fn gt(self, o: Self) -> Self {
                Self(i16x8_gt(self.0, o.0))
            }
            #[inline(always)]
            pub fn and(self, o: Self) -> Self {
                Self(v128_and(self.0, o.0))
            }
            #[inline(always)]
            pub fn or(self, o: Self) -> Self {
                Self(v128_or(self.0, o.0))
            }
            #[inline(always)]
            pub fn andnot(self, excl: Self) -> Self {
                Self(v128_andnot(self.0, excl.0))
            }
            #[inline(always)]
            pub fn select(mask: Self, new: Self, old: Self) -> Self {
                Self(v128_bitselect(new.0, old.0, mask.0))
            }
            #[inline(always)]
            pub fn any(self) -> bool {
                v128_any_true(self.0)
            }
            #[inline(always)]
            pub fn transpose8(r: [Self; 8]) -> [Self; 8] {
                let a0 = u16x8_shuffle::<0, 8, 2, 10, 4, 12, 6, 14>(r[0].0, r[1].0);
                let a1 = u16x8_shuffle::<1, 9, 3, 11, 5, 13, 7, 15>(r[0].0, r[1].0);
                let a2 = u16x8_shuffle::<0, 8, 2, 10, 4, 12, 6, 14>(r[2].0, r[3].0);
                let a3 = u16x8_shuffle::<1, 9, 3, 11, 5, 13, 7, 15>(r[2].0, r[3].0);
                let a4 = u16x8_shuffle::<0, 8, 2, 10, 4, 12, 6, 14>(r[4].0, r[5].0);
                let a5 = u16x8_shuffle::<1, 9, 3, 11, 5, 13, 7, 15>(r[4].0, r[5].0);
                let a6 = u16x8_shuffle::<0, 8, 2, 10, 4, 12, 6, 14>(r[6].0, r[7].0);
                let a7 = u16x8_shuffle::<1, 9, 3, 11, 5, 13, 7, 15>(r[6].0, r[7].0);

                let b0 = u32x4_shuffle::<0, 4, 2, 6>(a0, a2);
                let b2 = u32x4_shuffle::<1, 5, 3, 7>(a0, a2);
                let b1 = u32x4_shuffle::<0, 4, 2, 6>(a1, a3);
                let b3 = u32x4_shuffle::<1, 5, 3, 7>(a1, a3);
                let b4 = u32x4_shuffle::<0, 4, 2, 6>(a4, a6);
                let b6 = u32x4_shuffle::<1, 5, 3, 7>(a4, a6);
                let b5 = u32x4_shuffle::<0, 4, 2, 6>(a5, a7);
                let b7 = u32x4_shuffle::<1, 5, 3, 7>(a5, a7);

                [
                    Self(u64x2_shuffle::<0, 2>(b0, b4)),
                    Self(u64x2_shuffle::<0, 2>(b1, b5)),
                    Self(u64x2_shuffle::<0, 2>(b2, b6)),
                    Self(u64x2_shuffle::<0, 2>(b3, b7)),
                    Self(u64x2_shuffle::<1, 3>(b0, b4)),
                    Self(u64x2_shuffle::<1, 3>(b1, b5)),
                    Self(u64x2_shuffle::<1, 3>(b2, b6)),
                    Self(u64x2_shuffle::<1, 3>(b3, b7)),
                ]
            }
        }
    }

    use vec16::V16;

    /// Core 8-lane §8.8.5 filter over `p3..q3` (and lazily `p7..p4, q4..q7` only if `wide16` fires).
    #[inline(always)]
    fn filter_chunk8<const HORIZONTAL: bool, const FOUR_LANES: bool>(
        base: *mut u16,
        q0: usize,
        step: usize,
        along: usize,
        strength: FilterStrength,
        filter_size: u8,
        bit_depth: u8,
    ) {
        let wide16 = filter_size >= TX_16X16;
        let shift = u32::from(bit_depth - 8);
        let limit = V16::splat((strength.limit as i16) << shift);
        let blimit = V16::splat((strength.blimit as i16) << shift);
        let thresh = V16::splat((strength.thresh as i16) << shift);
        let one = V16::splat(1i16 << shift);
        let half = 1i16 << (bit_depth - 1);
        let lo_c = V16::splat(-half);
        let hi_c = V16::splat(half - 1);
        let offset = V16::splat(0x80i16 << shift);

        // 1. Load only the inner 8 taps (p3..q3 = k in 4..12) first!
        let inner: [V16; 8] = if HORIZONTAL {
            core::array::from_fn(|idx| {
                let at = (q0 + (idx + 4) * step) - 8 * step;
                unsafe {
                    if FOUR_LANES {
                        V16::load4(base.add(at))
                    } else {
                        V16::load8(base.add(at))
                    }
                }
            })
        } else {
            let rows: [V16; 8] = core::array::from_fn(|l| {
                if FOUR_LANES && l >= 4 {
                    V16::splat(0)
                } else {
                    let at = q0 + l * along - 4;
                    unsafe { V16::load8(base.add(at)) }
                }
            });
            V16::transpose8(rows)
        };

        let [p3, p2, p1, p0, q0v, q1, q2, q3] = inner;

        let mut mask = p3
            .abs_diff(p2)
            .le(limit)
            .and(p2.abs_diff(p1).le(limit))
            .and(p1.abs_diff(p0).le(limit))
            .and(q1.abs_diff(q0v).le(limit))
            .and(q2.abs_diff(q1).le(limit))
            .and(q3.abs_diff(q2).le(limit))
            .and(
                p0.abs_diff(q0v)
                    .shl1()
                    .add(p1.abs_diff(q1).shr1())
                    .le(blimit),
            );
        if FOUR_LANES {
            mask = mask.and(V16::lane_mask4());
        }
        // Early-exit when no lane crosses the edge threshold (~45% of edges)!
        if !mask.any() {
            return;
        }

        let m_hev = p1.abs_diff(p0).gt(thresh).or(q1.abs_diff(q0v).gt(thresh));
        let (m_narrow, m_wide8, m_wide16, outer_lo, outer_hi) = if filter_size == TX_4X4 {
            (
                mask,
                V16::splat(0),
                V16::splat(0),
                [V16::splat(0); 4],
                [V16::splat(0); 4],
            )
        } else {
            let flat = p1
                .abs_diff(p0)
                .le(one)
                .and(q1.abs_diff(q0v).le(one))
                .and(p2.abs_diff(p0).le(one))
                .and(q2.abs_diff(q0v).le(one))
                .and(p3.abs_diff(p0).le(one))
                .and(q3.abs_diff(q0v).le(one));
            let wide = mask.and(flat);
            let mn = mask.andnot(wide);
            if !wide16 || !wide.any() {
                (
                    mn,
                    wide,
                    V16::splat(0),
                    [V16::splat(0); 4],
                    [V16::splat(0); 4],
                )
            } else {
                // Load outer p7..p4 and q4..q7 ONLY when wide16 AND at least one lane is flat!
                let (olo, ohi): ([V16; 4], [V16; 4]) = if HORIZONTAL {
                    let lo = core::array::from_fn(|k| {
                        let at = (q0 + k * step) - 8 * step;
                        unsafe {
                            if FOUR_LANES {
                                V16::load4(base.add(at))
                            } else {
                                V16::load8(base.add(at))
                            }
                        }
                    });
                    let hi = core::array::from_fn(|k| {
                        let at = (q0 + (k + 12) * step) - 8 * step;
                        unsafe {
                            if FOUR_LANES {
                                V16::load4(base.add(at))
                            } else {
                                V16::load8(base.add(at))
                            }
                        }
                    });
                    (lo, hi)
                } else {
                    let rows_lo: [V16; 8] = core::array::from_fn(|l| {
                        if FOUR_LANES && l >= 4 {
                            V16::splat(0)
                        } else {
                            let at = q0 + l * along - 8;
                            unsafe { V16::load4(base.add(at)) }
                        }
                    });
                    let rows_hi: [V16; 8] = core::array::from_fn(|l| {
                        if FOUR_LANES && l >= 4 {
                            V16::splat(0)
                        } else {
                            let at = q0 + l * along + 4;
                            unsafe { V16::load4(base.add(at)) }
                        }
                    });
                    let tlo = V16::transpose8(rows_lo);
                    let thi = V16::transpose8(rows_hi);
                    (
                        [tlo[0], tlo[1], tlo[2], tlo[3]],
                        [thi[0], thi[1], thi[2], thi[3]],
                    )
                };
                let flat2 = olo[0]
                    .abs_diff(p0)
                    .le(one)
                    .and(ohi[3].abs_diff(q0v).le(one))
                    .and(olo[1].abs_diff(p0).le(one))
                    .and(ohi[2].abs_diff(q0v).le(one))
                    .and(olo[2].abs_diff(p0).le(one))
                    .and(ohi[1].abs_diff(q0v).le(one))
                    .and(olo[3].abs_diff(p0).le(one))
                    .and(ohi[0].abs_diff(q0v).le(one));
                let mw16 = wide.and(flat2);
                let mw8 = wide.andnot(flat2);
                (mn, mw8, mw16, olo, ohi)
            }
        };

        let t = [
            outer_lo[0],
            outer_lo[1],
            outer_lo[2],
            outer_lo[3],
            p3,
            p2,
            p1,
            p0,
            q0v,
            q1,
            q2,
            q3,
            outer_hi[0],
            outer_hi[1],
            outer_hi[2],
            outer_hi[3],
        ];
        let mut o = t;

        if m_narrow.any() {
            let c = |v: V16| v.clamp(lo_c, hi_c);
            let ps1 = p1.sub(offset);
            let ps0 = p0.sub(offset);
            let qs0 = q0v.sub(offset);
            let qs1 = q1.sub(offset);
            let f0 = c(ps1.sub(qs1)).and(m_hev);
            let f = c(f0.add(qs0.sub(ps0).mul3()));
            let f1 = c(f.add(V16::splat(4))).shr3();
            let f2 = c(f.add(V16::splat(3))).shr3();
            o[8] = V16::select(m_narrow, c(qs0.sub(f1)).add(offset), o[8]);
            o[7] = V16::select(m_narrow, c(ps0.add(f2)).add(offset), o[7]);
            let fs = f1.add(V16::splat(1)).shr1();
            let m1 = m_narrow.andnot(m_hev);
            o[9] = V16::select(m1, c(qs1.sub(fs)).add(offset), o[9]);
            o[6] = V16::select(m1, c(ps1.add(fs)).add(offset), o[6]);
        }

        let has_w8 = m_wide8.any();
        if has_w8 {
            let get = |k: i32| t[(k.clamp(-4, 3) + 8) as usize];
            let mut window = (-3..=3).fold(V16::splat(0), |acc, j| acc.add(get(-3 + j)));
            let four = V16::splat(4);
            for i in -3..3 {
                let idx = (i + 8) as usize;
                let nv = get(i).add(window).add(four).shr3();
                o[idx] = V16::select(m_wide8, nv, o[idx]);
                window = window.add(get(i + 4)).sub(get(i - 3));
            }
        }

        let has_w16 = m_wide16.any();
        if has_w16 {
            let get = |k: i32| t[(k.clamp(-8, 7) + 8) as usize];
            let mut window = (-7..=7).fold(V16::splat(0), |acc, j| acc.add(get(-7 + j)));
            let eight = V16::splat(8);
            for i in -7..7 {
                let idx = (i + 8) as usize;
                let nv = get(i).add(window).add(eight).shr4();
                o[idx] = V16::select(m_wide16, nv, o[idx]);
                window = window.add(get(i + 8)).sub(get(i - 7));
            }
        }

        // Scatter only the taps actually modified!
        if HORIZONTAL {
            let (lo, hi) = if has_w16 {
                (1, 15)
            } else if has_w8 {
                (5, 11)
            } else {
                (6, 10)
            };
            for k in lo..hi {
                let at = (q0 + k * step) - 8 * step;
                unsafe {
                    if FOUR_LANES {
                        o[k].store4(base.add(at));
                    } else {
                        o[k].store8(base.add(at));
                    }
                }
            }
        } else {
            let rows_count = if FOUR_LANES { 4 } else { 8 };
            if !has_w16 {
                // Only taps 4..12 (`o[4..12]`) could have changed: single 8x8 untranspose!
                let rows = V16::transpose8([o[4], o[5], o[6], o[7], o[8], o[9], o[10], o[11]]);
                for l in 0..rows_count {
                    let at = q0 + l * along - 4;
                    unsafe { rows[l].store8(base.add(at)) };
                }
            } else {
                let rows_lo = V16::transpose8([o[0], o[1], o[2], o[3], o[4], o[5], o[6], o[7]]);
                let rows_hi =
                    V16::transpose8([o[8], o[9], o[10], o[11], o[12], o[13], o[14], o[15]]);
                for l in 0..rows_count {
                    let at = q0 + l * along - 8;
                    unsafe {
                        rows_lo[l].store8(base.add(at));
                        rows_hi[l].store8(base.add(at + 8));
                    }
                }
            }
        }
    }

    pub fn filter_lanes_u16(
        data: &mut [u16],
        q0: usize,
        step: usize,
        along: usize,
        n: usize,
        strength: FilterStrength,
        filter_size: u8,
        bit_depth: u8,
    ) -> bool {
        if bit_depth > 10 {
            return false;
        }
        let base = data.as_mut_ptr();
        match (along == 1, n) {
            (true, 8) => filter_chunk8::<true, false>(
                base,
                q0,
                step,
                along,
                strength,
                filter_size,
                bit_depth,
            ),
            (true, 16) => {
                filter_chunk8::<true, false>(
                    base,
                    q0,
                    step,
                    along,
                    strength,
                    filter_size,
                    bit_depth,
                );
                filter_chunk8::<true, false>(
                    base,
                    q0 + 8,
                    step,
                    along,
                    strength,
                    filter_size,
                    bit_depth,
                );
            }
            (true, 4) => {
                filter_chunk8::<true, true>(base, q0, step, along, strength, filter_size, bit_depth)
            }
            (true, 12) => {
                filter_chunk8::<true, false>(
                    base,
                    q0,
                    step,
                    along,
                    strength,
                    filter_size,
                    bit_depth,
                );
                filter_chunk8::<true, true>(
                    base,
                    q0 + 8,
                    step,
                    along,
                    strength,
                    filter_size,
                    bit_depth,
                );
            }
            (false, 8) => filter_chunk8::<false, false>(
                base,
                q0,
                step,
                along,
                strength,
                filter_size,
                bit_depth,
            ),
            (false, 16) => {
                filter_chunk8::<false, false>(
                    base,
                    q0,
                    step,
                    along,
                    strength,
                    filter_size,
                    bit_depth,
                );
                filter_chunk8::<false, false>(
                    base,
                    q0 + 8 * along,
                    step,
                    along,
                    strength,
                    filter_size,
                    bit_depth,
                );
            }
            (false, 4) => filter_chunk8::<false, true>(
                base,
                q0,
                step,
                along,
                strength,
                filter_size,
                bit_depth,
            ),
            (false, 12) => {
                filter_chunk8::<false, false>(
                    base,
                    q0,
                    step,
                    along,
                    strength,
                    filter_size,
                    bit_depth,
                );
                filter_chunk8::<false, true>(
                    base,
                    q0 + 8 * along,
                    step,
                    along,
                    strength,
                    filter_size,
                    bit_depth,
                );
            }
            _ => return false,
        }
        true
    }

    // -----------------------------------------------------------------------
    // §8.5.2.4 8-tap subpel convolution over u16 reference / i16 intermediate
    // -----------------------------------------------------------------------

    #[inline(always)]
    fn tap_bounds(taps: &[i32; 8]) -> (usize, usize) {
        let s = taps.iter().position(|&t| t != 0).unwrap_or(0);
        let e = taps.iter().rposition(|&t| t != 0).map_or(0, |p| p + 1);
        (s, e)
    }

    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    pub fn convolve_h_u16_to_i16(
        row: &[u16],
        taps: &[i32; 8],
        max: i32,
        dst: &mut [i16],
        w: usize,
    ) {
        use core::arch::aarch64::*;
        let (ts, te) = tap_bounds(taps);
        let end8 = w & !7;
        unsafe {
            let round = vdupq_n_s32(64);
            let top = vdup_n_u16(max as u16);
            let mut x = 0;
            while x < end8 {
                let p = row.as_ptr().add(x);
                let (mut lo, mut hi) = (round, round);
                for k in ts..te {
                    let t = taps[k] as i16;
                    let v = vreinterpretq_s16_u16(vld1q_u16(p.add(k)));
                    lo = vmlal_n_s16(lo, vget_low_s16(v), t);
                    hi = vmlal_high_n_s16(hi, v, t);
                }
                let r_lo = vmin_u16(vqmovun_s32(vshrq_n_s32::<7>(lo)), top);
                let r_hi = vmin_u16(vqmovun_s32(vshrq_n_s32::<7>(hi)), top);
                vst1q_s16(
                    dst.as_mut_ptr().add(x),
                    vreinterpretq_s16_u16(vcombine_u16(r_lo, r_hi)),
                );
                x += 8;
            }
            if end8 + 4 <= w {
                let p = row.as_ptr().add(end8);
                let mut acc = round;
                for k in ts..te {
                    let v = vreinterpret_s16_u16(vld1_u16(p.add(k)));
                    acc = vmlal_n_s16(acc, v, taps[k] as i16);
                }
                let r = vmin_u16(vqmovun_s32(vshrq_n_s32::<7>(acc)), top);
                vst1_s16(dst.as_mut_ptr().add(end8), vreinterpret_s16_u16(r));
                x += 4;
            }
            for c in x..w {
                let mut s = 0i32;
                for k in ts..te {
                    s += taps[k] * i32::from(row[c + k]);
                }
                dst[c] = ((s + 64) >> 7).clamp(0, max) as i16;
            }
        }
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    #[inline(always)]
    pub fn convolve_h_u16_to_i16(
        row: &[u16],
        taps: &[i32; 8],
        max: i32,
        dst: &mut [i16],
        w: usize,
    ) {
        use core::arch::wasm32::*;
        let (ts, te) = tap_bounds(taps);
        let end8 = w & !7;
        unsafe {
            let tv: [v128; 8] = core::array::from_fn(|k| i16x8_splat(taps[k] as i16));
            let round = i32x4_splat(64);
            let zero = i32x4_splat(0);
            let top = i32x4_splat(max);
            let mut x = 0;
            while x < end8 {
                let p = row.as_ptr().add(x);
                let (mut lo, mut hi) = (round, round);
                for k in ts..te {
                    let v = v128_load(p.add(k).cast());
                    lo = i32x4_add(lo, i32x4_extmul_low_i16x8(v, tv[k]));
                    hi = i32x4_add(hi, i32x4_extmul_high_i16x8(v, tv[k]));
                }
                let r_lo = i32x4_min(i32x4_max(i32x4_shr(lo, 7), zero), top);
                let r_hi = i32x4_min(i32x4_max(i32x4_shr(hi, 7), zero), top);
                v128_store(
                    dst.as_mut_ptr().add(x).cast(),
                    i16x8_narrow_i32x4(r_lo, r_hi),
                );
                x += 8;
            }
            if end8 + 4 <= w {
                let p = row.as_ptr().add(end8);
                let mut acc = round;
                for k in ts..te {
                    let v = v128_load64_zero(p.add(k).cast());
                    acc = i32x4_add(acc, i32x4_extmul_low_i16x8(v, tv[k]));
                }
                let r = i32x4_min(i32x4_max(i32x4_shr(acc, 7), zero), top);
                v128_store64_lane::<0>(i16x8_narrow_i32x4(r, r), dst.as_mut_ptr().add(end8).cast());
                x += 4;
            }
            for c in x..w {
                let mut s = 0i32;
                for k in ts..te {
                    s += taps[k] * i32::from(row[c + k]);
                }
                dst[c] = ((s + 64) >> 7).clamp(0, max) as i16;
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    pub fn convolve_v_i16_to_u16(
        rows: [&[i16]; 8],
        taps: &[i32; 8],
        max: i32,
        dst: &mut [u16],
        w: usize,
    ) {
        use core::arch::aarch64::*;
        let (ts, te) = tap_bounds(taps);
        let end8 = w & !7;
        unsafe {
            let round = vdupq_n_s32(64);
            let top = vdup_n_u16(max as u16);
            let mut x = 0;
            while x < end8 {
                let (mut lo, mut hi) = (round, round);
                for k in ts..te {
                    let t = taps[k] as i16;
                    let v = vld1q_s16(rows[k].as_ptr().add(x));
                    lo = vmlal_n_s16(lo, vget_low_s16(v), t);
                    hi = vmlal_high_n_s16(hi, v, t);
                }
                let r_lo = vmin_u16(vqmovun_s32(vshrq_n_s32::<7>(lo)), top);
                let r_hi = vmin_u16(vqmovun_s32(vshrq_n_s32::<7>(hi)), top);
                vst1q_u16(dst.as_mut_ptr().add(x), vcombine_u16(r_lo, r_hi));
                x += 8;
            }
            if end8 + 4 <= w {
                let mut acc = round;
                for k in ts..te {
                    let v = vld1_s16(rows[k].as_ptr().add(end8));
                    acc = vmlal_n_s16(acc, v, taps[k] as i16);
                }
                let r = vmin_u16(vqmovun_s32(vshrq_n_s32::<7>(acc)), top);
                vst1_u16(dst.as_mut_ptr().add(end8), r);
                x += 4;
            }
            for c in x..w {
                let mut s = 0i32;
                for k in ts..te {
                    s += taps[k] * i32::from(rows[k][c]);
                }
                dst[c] = ((s + 64) >> 7).clamp(0, max) as u16;
            }
        }
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    #[inline(always)]
    pub fn convolve_v_i16_to_u16(
        rows: [&[i16]; 8],
        taps: &[i32; 8],
        max: i32,
        dst: &mut [u16],
        w: usize,
    ) {
        use core::arch::wasm32::*;
        let (ts, te) = tap_bounds(taps);
        let end8 = w & !7;
        unsafe {
            let tv: [v128; 8] = core::array::from_fn(|k| i16x8_splat(taps[k] as i16));
            let round = i32x4_splat(64);
            let zero = i32x4_splat(0);
            let top = i32x4_splat(max);
            let mut x = 0;
            while x < end8 {
                let (mut lo, mut hi) = (round, round);
                for k in ts..te {
                    let v = v128_load(rows[k].as_ptr().add(x).cast());
                    lo = i32x4_add(lo, i32x4_extmul_low_i16x8(v, tv[k]));
                    hi = i32x4_add(hi, i32x4_extmul_high_i16x8(v, tv[k]));
                }
                let r_lo = i32x4_min(i32x4_max(i32x4_shr(lo, 7), zero), top);
                let r_hi = i32x4_min(i32x4_max(i32x4_shr(hi, 7), zero), top);
                v128_store(
                    dst.as_mut_ptr().add(x).cast(),
                    i16x8_narrow_i32x4(r_lo, r_hi),
                );
                x += 8;
            }
            if end8 + 4 <= w {
                let mut acc = round;
                for k in ts..te {
                    let v = v128_load64_zero(rows[k].as_ptr().add(end8).cast());
                    acc = i32x4_add(acc, i32x4_extmul_low_i16x8(v, tv[k]));
                }
                let r = i32x4_min(i32x4_max(i32x4_shr(acc, 7), zero), top);
                v128_store64_lane::<0>(i16x8_narrow_i32x4(r, r), dst.as_mut_ptr().add(end8).cast());
                x += 4;
            }
            for c in x..w {
                let mut s = 0i32;
                for k in ts..te {
                    s += taps[k] * i32::from(rows[k][c]);
                }
                dst[c] = ((s + 64) >> 7).clamp(0, max) as u16;
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    pub fn convolve_v_i16_to_i32(
        rows: [&[i16]; 8],
        taps: &[i32; 8],
        max: i32,
        dst: &mut [i32],
        w: usize,
    ) {
        use core::arch::aarch64::*;
        let (ts, te) = tap_bounds(taps);
        let end8 = w & !7;
        unsafe {
            let round = vdupq_n_s32(64);
            let zero = vdupq_n_s32(0);
            let top = vdupq_n_s32(max);
            let mut x = 0;
            while x < end8 {
                let (mut lo, mut hi) = (round, round);
                for k in ts..te {
                    let t = taps[k] as i16;
                    let v = vld1q_s16(rows[k].as_ptr().add(x));
                    lo = vmlal_n_s16(lo, vget_low_s16(v), t);
                    hi = vmlal_high_n_s16(hi, v, t);
                }
                vst1q_s32(
                    dst.as_mut_ptr().add(x),
                    vminq_s32(vmaxq_s32(vshrq_n_s32::<7>(lo), zero), top),
                );
                vst1q_s32(
                    dst.as_mut_ptr().add(x + 4),
                    vminq_s32(vmaxq_s32(vshrq_n_s32::<7>(hi), zero), top),
                );
                x += 8;
            }
            if end8 + 4 <= w {
                let mut acc = round;
                for k in ts..te {
                    let v = vld1_s16(rows[k].as_ptr().add(end8));
                    acc = vmlal_n_s16(acc, v, taps[k] as i16);
                }
                vst1q_s32(
                    dst.as_mut_ptr().add(end8),
                    vminq_s32(vmaxq_s32(vshrq_n_s32::<7>(acc), zero), top),
                );
                x += 4;
            }
            for c in x..w {
                let mut s = 0i32;
                for k in ts..te {
                    s += taps[k] * i32::from(rows[k][c]);
                }
                dst[c] = ((s + 64) >> 7).clamp(0, max);
            }
        }
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    #[inline(always)]
    pub fn convolve_v_i16_to_i32(
        rows: [&[i16]; 8],
        taps: &[i32; 8],
        max: i32,
        dst: &mut [i32],
        w: usize,
    ) {
        use core::arch::wasm32::*;
        let (ts, te) = tap_bounds(taps);
        let end8 = w & !7;
        unsafe {
            let tv: [v128; 8] = core::array::from_fn(|k| i16x8_splat(taps[k] as i16));
            let round = i32x4_splat(64);
            let zero = i32x4_splat(0);
            let top = i32x4_splat(max);
            let mut x = 0;
            while x < end8 {
                let (mut lo, mut hi) = (round, round);
                for k in ts..te {
                    let v = v128_load(rows[k].as_ptr().add(x).cast());
                    lo = i32x4_add(lo, i32x4_extmul_low_i16x8(v, tv[k]));
                    hi = i32x4_add(hi, i32x4_extmul_high_i16x8(v, tv[k]));
                }
                v128_store(
                    dst.as_mut_ptr().add(x).cast(),
                    i32x4_min(i32x4_max(i32x4_shr(lo, 7), zero), top),
                );
                v128_store(
                    dst.as_mut_ptr().add(x + 4).cast(),
                    i32x4_min(i32x4_max(i32x4_shr(hi, 7), zero), top),
                );
                x += 8;
            }
            if end8 + 4 <= w {
                let mut acc = round;
                for k in ts..te {
                    let v = v128_load64_zero(rows[k].as_ptr().add(end8).cast());
                    acc = i32x4_add(acc, i32x4_extmul_low_i16x8(v, tv[k]));
                }
                v128_store(
                    dst.as_mut_ptr().add(end8).cast(),
                    i32x4_min(i32x4_max(i32x4_shr(acc, 7), zero), top),
                );
                x += 4;
            }
            for c in x..w {
                let mut s = 0i32;
                for k in ts..te {
                    s += taps[k] * i32::from(rows[k][c]);
                }
                dst[c] = ((s + 64) >> 7).clamp(0, max);
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    pub fn avg_round2_u16_i32(dst: &mut [u16], src: &[i32]) {
        use core::arch::aarch64::*;
        let n = dst.len();
        let mut i = 0;
        unsafe {
            let one = vdupq_n_s32(1);
            while i + 8 <= n {
                let cur = vld1q_u16(dst.as_ptr().add(i));
                let lo = vshrq_n_s32::<1>(vaddq_s32(
                    vaddq_s32(
                        vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(cur))),
                        vld1q_s32(src.as_ptr().add(i)),
                    ),
                    one,
                ));
                let hi = vshrq_n_s32::<1>(vaddq_s32(
                    vaddq_s32(
                        vreinterpretq_s32_u32(vmovl_high_u16(cur)),
                        vld1q_s32(src.as_ptr().add(i + 4)),
                    ),
                    one,
                ));
                vst1q_u16(
                    dst.as_mut_ptr().add(i),
                    vcombine_u16(vqmovun_s32(lo), vqmovun_s32(hi)),
                );
                i += 8;
            }
        }
        for (o, &b) in dst[i..].iter_mut().zip(&src[i..n]) {
            *o = ((i32::from(*o) + b + 1) >> 1) as u16;
        }
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    pub fn avg_round2_u16_i32(dst: &mut [u16], src: &[i32]) {
        use core::arch::wasm32::*;
        let n = dst.len();
        let mut i = 0;
        unsafe {
            let one = i32x4_splat(1);
            while i + 8 <= n {
                let cur = v128_load(dst.as_ptr().add(i).cast());
                let lo = i32x4_shr(
                    i32x4_add(
                        i32x4_add(
                            u32x4_extend_low_u16x8(cur),
                            v128_load(src.as_ptr().add(i).cast()),
                        ),
                        one,
                    ),
                    1,
                );
                let hi = i32x4_shr(
                    i32x4_add(
                        i32x4_add(
                            u32x4_extend_high_u16x8(cur),
                            v128_load(src.as_ptr().add(i + 4).cast()),
                        ),
                        one,
                    ),
                    1,
                );
                v128_store(dst.as_mut_ptr().add(i).cast(), i16x8_narrow_i32x4(lo, hi));
                i += 8;
            }
        }
        for (o, &b) in dst[i..].iter_mut().zip(&src[i..n]) {
            *o = ((i32::from(*o) + b + 1) >> 1) as u16;
        }
    }
}

#[cfg(any(
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
pub(crate) use imp::*;
