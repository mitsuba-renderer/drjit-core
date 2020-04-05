define internal void @ek.scatter_add_v1f32(i8* nocapture, <1 x float>, <1 x i32>) local_unnamed_addr #2 {
L3:
  %r4 = extractelement <1 x float> %1, i32 0
  %r5 = bitcast i8* %0 to float*
  %r6 = extractelement <1 x i32> %2, i32 0
  %r7 = sext i32 %r6 to i64
  %r8 = getelementptr inbounds float, float* %r5, i64 %r7
  %r9 = load float, float* %r8, align 4
  %r10 = fadd float %r4, %r9
  store float %r10, float* %r8, align 4
  ret void
}

define internal void @ek.masked_scatter_add_v1f32(i8* nocapture, <1 x float>, <1 x i32>, <1 x i1>) local_unnamed_addr #2 {
L4:
  %r3 = extractelement <1 x i1> %3, i32 0
  br i1 %r3, label %L5, label %L13

L5:
  %r6 = extractelement <1 x float> %1, i32 0
  %r7 = bitcast i8* %0 to float*
  %r8 = extractelement <1 x i32> %2, i32 0
  %r9 = sext i32 %r8 to i64
  %r10 = getelementptr inbounds float, float* %r7, i64 %r9
  %r11 = load float, float* %r10, align 4
  %r12 = fadd float %r6, %r11
  store float %r12, float* %r10, align 4
  br label %L13

L13:
  ret void
}

define internal void @ek.scatter_add_v1i32(i8* nocapture, <1 x i32>, <1 x i32>) local_unnamed_addr #2 {
L3:
  %r4 = extractelement <1 x i32> %1, i32 0
  %r5 = bitcast i8* %0 to i32*
  %r6 = extractelement <1 x i32> %2, i32 0
  %r7 = sext i32 %r6 to i64
  %r8 = getelementptr inbounds i32, i32* %r5, i64 %r7
  %r9 = load i32, i32* %r8, align 4
  %r10 = add nsw i32 %r9, %r4
  store i32 %r10, i32* %r8, align 4
  ret void
}

define internal void @ek.masked_scatter_add_v1i32(i8* nocapture, <1 x i32>, <1 x i32>, <1 x i1>) local_unnamed_addr #2 {
L4:
  %r3 = extractelement <1 x i1> %3, i32 0
  br i1 %r3, label %L5, label %L13

L5:
  %r6 = extractelement <1 x i32> %1, i32 0
  %r7 = bitcast i8* %0 to i32*
  %r8 = extractelement <1 x i32> %2, i32 0
  %r9 = sext i32 %r8 to i64
  %r10 = getelementptr inbounds i32, i32* %r7, i64 %r9
  %r11 = load i32, i32* %r10, align 4
  %r12 = add nsw i32 %r11, %r6
  store i32 %r12, i32* %r10, align 4
  br label %L13

L13:
  ret void
}

define internal void @ek.scatter_add_v1f64(i8* nocapture, <1 x double>, <1 x i64>) local_unnamed_addr #2 {
L3:
  %r4 = extractelement <1 x double> %1, i32 0
  %r5 = bitcast i8* %0 to double*
  %r6 = extractelement <1 x i64> %2, i32 0
  %r7 = getelementptr inbounds double, double* %r5, i64 %r6
  %r8 = load double, double* %r7, align 8
  %r9 = fadd double %r4, %r8
  store double %r9, double* %r7, align 8
  ret void
}

define internal void @ek.masked_scatter_add_v1f64(i8* nocapture, <1 x double>, <1 x i64>, <1 x i1>) local_unnamed_addr #2 {
L4:
  %r3 = extractelement <1 x i1> %3, i32 0
  br i1 %r3, label %L5, label %L12

L5:
  %r6 = extractelement <1 x double> %1, i32 0
  %r7 = bitcast i8* %0 to double*
  %r8 = extractelement <1 x i64> %2, i32 0
  %r9 = getelementptr inbounds double, double* %r7, i64 %r8
  %r10 = load double, double* %r9, align 8
  %r11 = fadd double %r6, %r10
  store double %r11, double* %r9, align 8
  br label %L12

L12:
  ret void
}

define internal void @ek.scatter_add_v1i64(i8* nocapture, <1 x i64>, <1 x i64>) local_unnamed_addr #2 {
L3:
  %r4 = extractelement <1 x i64> %1, i32 0
  %r5 = bitcast i8* %0 to i64*
  %r6 = extractelement <1 x i64> %2, i32 0
  %r7 = getelementptr inbounds i64, i64* %r5, i64 %r6
  %r8 = load i64, i64* %r7, align 8
  %r9 = add nsw i64 %r8, %r4
  store i64 %r9, i64* %r7, align 8
  ret void
}

define internal void @ek.masked_scatter_add_v1i64(i8* nocapture, <1 x i64>, <1 x i64>, <1 x i1>) local_unnamed_addr #2 {
L4:
  %r3 = extractelement <1 x i1> %3, i32 0
  br i1 %r3, label %L5, label %L12

L5:
  %r6 = extractelement <1 x i64> %1, i32 0
  %r7 = bitcast i8* %0 to i64*
  %r8 = extractelement <1 x i64> %2, i32 0
  %r9 = getelementptr inbounds i64, i64* %r7, i64 %r8
  %r10 = load i64, i64* %r9, align 8
  %r11 = add nsw i64 %r10, %r6
  store i64 %r11, i64* %r9, align 8
  br label %L12

L12:
  ret void
}

define internal void @ek.scatter_add_v16f32(i8*, <16 x float>, <16 x i32>) local_unnamed_addr #1 {
L3:
  %r4 = tail call <16 x float> @llvm.x86.avx512.gather.dps.512(<16 x float> undef, i8* %0, <16 x i32> %2, i16 -1, i32 4)
  %r5 = tail call <16 x i32> @llvm.x86.avx512.mask.conflict.d.512(<16 x i32> %2, <16 x i32> zeroinitializer, i16 -1)
  %r6 = icmp ne <16 x i32> %r5, zeroinitializer
  %r7 = bitcast <16 x i1> %r6 to i16
  %r8 = icmp eq i16 %r7, 0
  br i1 %r8, label %L24, label %L9, !prof !{!"branch_weights", i32 2000, i32 1}

L9:
  %r10 = tail call <16 x i32> @llvm.ctlz.v16i32(<16 x i32> %r5, i1 false)
  %r11 = sub nsw <16 x i32> <i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31>, %r10
  br label %L12

L12:
  %r13 = phi <16 x i32> [ %r11, %L9 ], [ %r19, %L12 ]
  %r14 = phi <16 x i1> [ %r6, %L9 ], [ %r21, %L12 ]
  %r15 = phi <16 x float> [ %1, %L9 ], [ %r20, %L12 ]
  %r16 = tail call <16 x float> @llvm.x86.avx512.permvar.sf.512(<16 x float> %r15, <16 x i32> %r13)
  %r17 = select <16 x i1> %r14, <16 x float> %r16, <16 x float> zeroinitializer
  %r18 = tail call <16 x i32> @llvm.x86.avx512.permvar.si.512(<16 x i32> %r13, <16 x i32> %r13)
  %r19 = select <16 x i1> %r14, <16 x i32> %r18, <16 x i32> %r13
  %r20 = fadd <16 x float> %r15, %r17
  %r21 = icmp ne <16 x i32> %r19, <i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1>
  %r22 = bitcast <16 x i1> %r21 to i16
  %r23 = icmp eq i16 %r22, 0
  br i1 %r23, label %L24, label %L12

L24:
  %r25 = phi <16 x float> [ %1, %L3 ], [ %r20, %L12 ]
  %r26 = fadd <16 x float> %r4, %r25
  tail call void @llvm.x86.avx512.scatter.dps.512(i8* %0, i16 -1, <16 x i32> %2, <16 x float> %r26, i32 4)
  ret void
}

define internal void @ek.masked_scatter_add_v16f32(i8*, <16 x float>, <16 x i32>, <16 x i1>) local_unnamed_addr #1 {
L4:
  %r5 = bitcast <16 x i1> %3 to i16
  %r6 = tail call <16 x float> @llvm.x86.avx512.gather.dps.512(<16 x float> undef, i8* %0, <16 x i32> %2, i16 %r5, i32 4)
  %r7 = tail call <16 x i32> @llvm.x86.avx512.mask.conflict.d.512(<16 x i32> %2, <16 x i32> zeroinitializer, i16 -1)
  %r8 = zext i16 %r5 to i32
  %r9 = insertelement <16 x i32> undef, i32 %r8, i32 0
  %r10 = shufflevector <16 x i32> %r9, <16 x i32> undef, <16 x i32> zeroinitializer
  %r11 = and <16 x i32> %r7, %r10
  %r12 = icmp ne <16 x i32> %r11, zeroinitializer
  %r13 = bitcast <16 x i1> %r12 to i16
  %r14 = icmp eq i16 %r13, 0
  br i1 %r14, label %L31, label %L15, !prof !{!"branch_weights", i32 2000, i32 1}

L15:
  %r16 = tail call <16 x i32> @llvm.ctlz.v16i32(<16 x i32> %r11, i1 false)
  %r17 = sub nsw <16 x i32> <i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31>, %r16
  br label %L18

L18:
  %r19 = phi <16 x i32> [ %r17, %L15 ], [ %r25, %L18 ]
  %r20 = phi <16 x i1> [ %r12, %L15 ], [ %r28, %L18 ]
  %r21 = phi <16 x float> [ %1, %L15 ], [ %r26, %L18 ]
  %r22 = tail call <16 x float> @llvm.x86.avx512.permvar.sf.512(<16 x float> %r21, <16 x i32> %r19)
  %r23 = select <16 x i1> %r20, <16 x float> %r22, <16 x float> zeroinitializer
  %r24 = tail call <16 x i32> @llvm.x86.avx512.permvar.si.512(<16 x i32> %r19, <16 x i32> %r19)
  %r25 = select <16 x i1> %r20, <16 x i32> %r24, <16 x i32> %r19
  %r26 = fadd <16 x float> %r21, %r23
  %r27 = icmp ne <16 x i32> %r25, <i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1>
  %r28 = and <16 x i1> %r27, %3
  %r29 = bitcast <16 x i1> %r28 to i16
  %r30 = icmp eq i16 %r29, 0
  br i1 %r30, label %L31, label %L18

L31:
  %r32 = phi <16 x float> [ %1, %L4 ], [ %r26, %L18 ]
  %r33 = fadd <16 x float> %r6, %r32
  tail call void @llvm.x86.avx512.scatter.dps.512(i8* %0, i16 %r5, <16 x i32> %2, <16 x float> %r33, i32 4)
  ret void
}

define internal void @ek.scatter_add_v16i32(i8*, <16 x i32>, <16 x i32>) local_unnamed_addr #1 {
L3:
  %r4 = tail call <16 x i32> @llvm.x86.avx512.gather.dpi.512(<16 x i32> undef, i8* %0, <16 x i32> %2, i16 -1, i32 4)
  %r5 = tail call <16 x i32> @llvm.x86.avx512.mask.conflict.d.512(<16 x i32> %2, <16 x i32> zeroinitializer, i16 -1)
  %r6 = icmp ne <16 x i32> %r5, zeroinitializer
  %r7 = bitcast <16 x i1> %r6 to i16
  %r8 = icmp eq i16 %r7, 0
  br i1 %r8, label %L24, label %L9, !prof !{!"branch_weights", i32 2000, i32 1}

L9:
  %r10 = tail call <16 x i32> @llvm.ctlz.v16i32(<16 x i32> %r5, i1 false)
  %r11 = sub nsw <16 x i32> <i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31>, %r10
  br label %L12

L12:
  %r13 = phi <16 x i32> [ %r11, %L9 ], [ %r19, %L12 ]
  %r14 = phi <16 x i1> [ %r6, %L9 ], [ %r21, %L12 ]
  %r15 = phi <16 x i32> [ %1, %L9 ], [ %r20, %L12 ]
  %r16 = tail call <16 x i32> @llvm.x86.avx512.permvar.si.512(<16 x i32> %r15, <16 x i32> %r13)
  %r17 = select <16 x i1> %r14, <16 x i32> %r16, <16 x i32> zeroinitializer
  %r18 = tail call <16 x i32> @llvm.x86.avx512.permvar.si.512(<16 x i32> %r13, <16 x i32> %r13)
  %r19 = select <16 x i1> %r14, <16 x i32> %r18, <16 x i32> %r13
  %r20 = add <16 x i32> %r17, %r15
  %r21 = icmp ne <16 x i32> %r19, <i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1>
  %r22 = bitcast <16 x i1> %r21 to i16
  %r23 = icmp eq i16 %r22, 0
  br i1 %r23, label %L24, label %L12

L24:
  %r25 = phi <16 x i32> [ %1, %L3 ], [ %r20, %L12 ]
  %r26 = add <16 x i32> %r25, %r4
  tail call void @llvm.x86.avx512.scatter.dpi.512(i8* %0, i16 -1, <16 x i32> %2, <16 x i32> %r26, i32 4)
  ret void
}

define internal void @ek.masked_scatter_add_v16i32(i8*, <16 x i32>, <16 x i32>, <16 x i1>) local_unnamed_addr #1 {
L4:
  %r5 = bitcast <16 x i1> %3 to i16
  %r6 = tail call <16 x i32> @llvm.x86.avx512.gather.dpi.512(<16 x i32> undef, i8* %0, <16 x i32> %2, i16 %r5, i32 4)
  %r7 = tail call <16 x i32> @llvm.x86.avx512.mask.conflict.d.512(<16 x i32> %2, <16 x i32> zeroinitializer, i16 -1)
  %r8 = zext i16 %r5 to i32
  %r9 = insertelement <16 x i32> undef, i32 %r8, i32 0
  %r10 = shufflevector <16 x i32> %r9, <16 x i32> undef, <16 x i32> zeroinitializer
  %r11 = and <16 x i32> %r7, %r10
  %r12 = icmp ne <16 x i32> %r11, zeroinitializer
  %r13 = bitcast <16 x i1> %r12 to i16
  %r14 = icmp eq i16 %r13, 0
  br i1 %r14, label %L31, label %L15, !prof !{!"branch_weights", i32 2000, i32 1}

L15:
  %r16 = tail call <16 x i32> @llvm.ctlz.v16i32(<16 x i32> %r11, i1 false)
  %r17 = sub nsw <16 x i32> <i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31>, %r16
  br label %L18

L18:
  %r19 = phi <16 x i32> [ %r17, %L15 ], [ %r25, %L18 ]
  %r20 = phi <16 x i1> [ %r12, %L15 ], [ %r28, %L18 ]
  %r21 = phi <16 x i32> [ %1, %L15 ], [ %r26, %L18 ]
  %r22 = tail call <16 x i32> @llvm.x86.avx512.permvar.si.512(<16 x i32> %r21, <16 x i32> %r19)
  %r23 = select <16 x i1> %r20, <16 x i32> %r22, <16 x i32> zeroinitializer
  %r24 = tail call <16 x i32> @llvm.x86.avx512.permvar.si.512(<16 x i32> %r19, <16 x i32> %r19)
  %r25 = select <16 x i1> %r20, <16 x i32> %r24, <16 x i32> %r19
  %r26 = add <16 x i32> %r23, %r21
  %r27 = icmp ne <16 x i32> %r25, <i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1>
  %r28 = and <16 x i1> %r27, %3
  %r29 = bitcast <16 x i1> %r28 to i16
  %r30 = icmp eq i16 %r29, 0
  br i1 %r30, label %L31, label %L18

L31:
  %r32 = phi <16 x i32> [ %1, %L4 ], [ %r26, %L18 ]
  %r33 = add <16 x i32> %r32, %r6
  tail call void @llvm.x86.avx512.scatter.dpi.512(i8* %0, i16 %r5, <16 x i32> %2, <16 x i32> %r33, i32 4)
  ret void
}

define internal void @ek.scatter_add_v8f64(i8*, <8 x double>, <8 x i64>) local_unnamed_addr #1 {
L3:
  %r4 = tail call <8 x double> @llvm.x86.avx512.gather.qpd.512(<8 x double> undef, i8* %0, <8 x i64> %2, i8 -1, i32 8)
  %r5 = tail call <8 x i64> @llvm.x86.avx512.mask.conflict.q.512(<8 x i64> %2, <8 x i64> zeroinitializer, i8 -1)
  %r6 = icmp ne <8 x i64> %r5, zeroinitializer
  %r7 = bitcast <8 x i1> %r6 to i8
  %r8 = icmp eq i8 %r7, 0
  br i1 %r8, label %L24, label %L9, !prof !{!"branch_weights", i32 2000, i32 1}

L9:
  %r10 = tail call <8 x i64> @llvm.ctlz.v8i64(<8 x i64> %r5, i1 false)
  %r11 = sub nsw <8 x i64> <i64 63, i64 63, i64 63, i64 63, i64 63, i64 63, i64 63, i64 63>, %r10
  br label %L12

L12:
  %r13 = phi <8 x i64> [ %r11, %L9 ], [ %r19, %L12 ]
  %r14 = phi <8 x i1> [ %r6, %L9 ], [ %r21, %L12 ]
  %r15 = phi <8 x double> [ %1, %L9 ], [ %r20, %L12 ]
  %r16 = tail call <8 x double> @llvm.x86.avx512.permvar.df.512(<8 x double> %r15, <8 x i64> %r13)
  %r17 = select <8 x i1> %r14, <8 x double> %r16, <8 x double> zeroinitializer
  %r18 = tail call <8 x i64> @llvm.x86.avx512.permvar.di.512(<8 x i64> %r13, <8 x i64> %r13)
  %r19 = select <8 x i1> %r14, <8 x i64> %r18, <8 x i64> %r13
  %r20 = fadd <8 x double> %r15, %r17
  %r21 = icmp ne <8 x i64> %r19, <i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1>
  %r22 = bitcast <8 x i1> %r21 to i8
  %r23 = icmp eq i8 %r22, 0
  br i1 %r23, label %L24, label %L12

L24:
  %r25 = phi <8 x double> [ %1, %L3 ], [ %r20, %L12 ]
  %r26 = fadd <8 x double> %r4, %r25
  tail call void @llvm.x86.avx512.scatter.qpd.512(i8* %0, i8 -1, <8 x i64> %2, <8 x double> %r26, i32 8)
  ret void
}

define internal void @ek.masked_scatter_add_v8f64(i8*, <8 x double>, <8 x i64>, <8 x i1>) local_unnamed_addr #1 {
L4:
  %r5 = bitcast <8 x i1> %3 to i8
  %r6 = tail call <8 x double> @llvm.x86.avx512.gather.qpd.512(<8 x double> undef, i8* %0, <8 x i64> %2, i8 %r5, i32 8)
  %r7 = tail call <8 x i64> @llvm.x86.avx512.mask.conflict.q.512(<8 x i64> %2, <8 x i64> zeroinitializer, i8 -1)
  %r8 = zext i8 %r5 to i64
  %r9 = insertelement <8 x i64> undef, i64 %r8, i32 0
  %r10 = shufflevector <8 x i64> %r9, <8 x i64> undef, <8 x i32> zeroinitializer
  %r11 = and <8 x i64> %r7, %r10
  %r12 = icmp ne <8 x i64> %r11, zeroinitializer
  %r13 = bitcast <8 x i1> %r12 to i8
  %r14 = icmp eq i8 %r13, 0
  br i1 %r14, label %L31, label %L15, !prof !{!"branch_weights", i32 2000, i32 1}

L15:
  %r16 = tail call <8 x i64> @llvm.ctlz.v8i64(<8 x i64> %r11, i1 false)
  %r17 = sub nsw <8 x i64> <i64 63, i64 63, i64 63, i64 63, i64 63, i64 63, i64 63, i64 63>, %r16
  br label %L18

L18:
  %r19 = phi <8 x i64> [ %r17, %L15 ], [ %r25, %L18 ]
  %r20 = phi <8 x i1> [ %r12, %L15 ], [ %r28, %L18 ]
  %r21 = phi <8 x double> [ %1, %L15 ], [ %r26, %L18 ]
  %r22 = tail call <8 x double> @llvm.x86.avx512.permvar.df.512(<8 x double> %r21, <8 x i64> %r19)
  %r23 = select <8 x i1> %r20, <8 x double> %r22, <8 x double> zeroinitializer
  %r24 = tail call <8 x i64> @llvm.x86.avx512.permvar.di.512(<8 x i64> %r19, <8 x i64> %r19)
  %r25 = select <8 x i1> %r20, <8 x i64> %r24, <8 x i64> %r19
  %r26 = fadd <8 x double> %r21, %r23
  %r27 = icmp ne <8 x i64> %r25, <i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1>
  %r28 = and <8 x i1> %r27, %3
  %r29 = bitcast <8 x i1> %r28 to i8
  %r30 = icmp eq i8 %r29, 0
  br i1 %r30, label %L31, label %L18

L31:
  %r32 = phi <8 x double> [ %1, %L4 ], [ %r26, %L18 ]
  %r33 = fadd <8 x double> %r6, %r32
  tail call void @llvm.x86.avx512.scatter.qpd.512(i8* %0, i8 %r5, <8 x i64> %2, <8 x double> %r33, i32 8)
  ret void
}

define internal void @ek.scatter_add_v8i64(i8*, <8 x i64>, <8 x i64>) local_unnamed_addr #1 {
L3:
  %r4 = tail call <8 x i64> @llvm.x86.avx512.gather.qpq.512(<8 x i64> undef, i8* %0, <8 x i64> %2, i8 -1, i32 8)
  %r5 = tail call <8 x i64> @llvm.x86.avx512.mask.conflict.q.512(<8 x i64> %2, <8 x i64> zeroinitializer, i8 -1)
  %r6 = icmp ne <8 x i64> %r5, zeroinitializer
  %r7 = bitcast <8 x i1> %r6 to i8
  %r8 = icmp eq i8 %r7, 0
  br i1 %r8, label %L24, label %L9, !prof !{!"branch_weights", i32 2000, i32 1}

L9:
  %r10 = tail call <8 x i64> @llvm.ctlz.v8i64(<8 x i64> %r5, i1 false)
  %r11 = sub nsw <8 x i64> <i64 63, i64 63, i64 63, i64 63, i64 63, i64 63, i64 63, i64 63>, %r10
  br label %L12

L12:
  %r13 = phi <8 x i64> [ %r11, %L9 ], [ %r19, %L12 ]
  %r14 = phi <8 x i1> [ %r6, %L9 ], [ %r21, %L12 ]
  %r15 = phi <8 x i64> [ %1, %L9 ], [ %r20, %L12 ]
  %r16 = tail call <8 x i64> @llvm.x86.avx512.permvar.di.512(<8 x i64> %r15, <8 x i64> %r13)
  %r17 = select <8 x i1> %r14, <8 x i64> %r16, <8 x i64> zeroinitializer
  %r18 = tail call <8 x i64> @llvm.x86.avx512.permvar.di.512(<8 x i64> %r13, <8 x i64> %r13)
  %r19 = select <8 x i1> %r14, <8 x i64> %r18, <8 x i64> %r13
  %r20 = add <8 x i64> %r17, %r15
  %r21 = icmp ne <8 x i64> %r19, <i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1>
  %r22 = bitcast <8 x i1> %r21 to i8
  %r23 = icmp eq i8 %r22, 0
  br i1 %r23, label %L24, label %L12

L24:
  %r25 = phi <8 x i64> [ %1, %L3 ], [ %r20, %L12 ]
  %r26 = add <8 x i64> %r25, %r4
  tail call void @llvm.x86.avx512.scatter.qpq.512(i8* %0, i8 -1, <8 x i64> %2, <8 x i64> %r26, i32 8)
  ret void
}

define internal void @ek.masked_scatter_add_v8i64(i8*, <8 x i64>, <8 x i64>, <8 x i1>) local_unnamed_addr #1 {
L4:
  %r5 = bitcast <8 x i1> %3 to i8
  %r6 = tail call <8 x i64> @llvm.x86.avx512.gather.qpq.512(<8 x i64> undef, i8* %0, <8 x i64> %2, i8 %r5, i32 8)
  %r7 = tail call <8 x i64> @llvm.x86.avx512.mask.conflict.q.512(<8 x i64> %2, <8 x i64> zeroinitializer, i8 -1)
  %r8 = zext i8 %r5 to i64
  %r9 = insertelement <8 x i64> undef, i64 %r8, i32 0
  %r10 = shufflevector <8 x i64> %r9, <8 x i64> undef, <8 x i32> zeroinitializer
  %r11 = and <8 x i64> %r7, %r10
  %r12 = icmp ne <8 x i64> %r11, zeroinitializer
  %r13 = bitcast <8 x i1> %r12 to i8
  %r14 = icmp eq i8 %r13, 0
  br i1 %r14, label %L31, label %L15, !prof !{!"branch_weights", i32 2000, i32 1}

L15:
  %r16 = tail call <8 x i64> @llvm.ctlz.v8i64(<8 x i64> %r11, i1 false)
  %r17 = sub nsw <8 x i64> <i64 63, i64 63, i64 63, i64 63, i64 63, i64 63, i64 63, i64 63>, %r16
  br label %L18

L18:
  %r19 = phi <8 x i64> [ %r17, %L15 ], [ %r25, %L18 ]
  %r20 = phi <8 x i1> [ %r12, %L15 ], [ %r28, %L18 ]
  %r21 = phi <8 x i64> [ %1, %L15 ], [ %r26, %L18 ]
  %r22 = tail call <8 x i64> @llvm.x86.avx512.permvar.di.512(<8 x i64> %r21, <8 x i64> %r19)
  %r23 = select <8 x i1> %r20, <8 x i64> %r22, <8 x i64> zeroinitializer
  %r24 = tail call <8 x i64> @llvm.x86.avx512.permvar.di.512(<8 x i64> %r19, <8 x i64> %r19)
  %r25 = select <8 x i1> %r20, <8 x i64> %r24, <8 x i64> %r19
  %r26 = add <8 x i64> %r23, %r21
  %r27 = icmp ne <8 x i64> %r25, <i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1>
  %r28 = and <8 x i1> %r27, %3
  %r29 = bitcast <8 x i1> %r28 to i8
  %r30 = icmp eq i8 %r29, 0
  br i1 %r30, label %L31, label %L18

L31:
  %r32 = phi <8 x i64> [ %1, %L4 ], [ %r26, %L18 ]
  %r33 = add <8 x i64> %r32, %r6
  tail call void @llvm.x86.avx512.scatter.qpq.512(i8* %0, i8 %r5, <8 x i64> %2, <8 x i64> %r33, i32 8)
  ret void
}

declare <16 x float> @llvm.x86.avx512.gather.dps.512(<16 x float>, i8*, <16 x i32>, i16, i32)
declare <16 x i32>   @llvm.x86.avx512.gather.dpi.512(<16 x i32>, i8*, <16 x i32>, i16, i32)
declare <8 x double> @llvm.x86.avx512.gather.qpd.512(<8 x double>, i8*, <8 x i64>, i8, i32)
declare <8 x i64>    @llvm.x86.avx512.gather.qpq.512(<8 x i64>, i8*, <8 x i64>, i8, i32)

declare void         @llvm.x86.avx512.scatter.dps.512(i8*, i16, <16 x i32>, <16 x float>, i32)
declare void         @llvm.x86.avx512.scatter.dpi.512(i8*, i16, <16 x i32>, <16 x i32>, i32)
declare void         @llvm.x86.avx512.scatter.qpd.512(i8*, i8, <8 x i64>, <8 x double>, i32)
declare void         @llvm.x86.avx512.scatter.qpq.512(i8*, i8, <8 x i64>, <8 x i64>, i32)

declare <16 x float> @llvm.x86.avx512.permvar.sf.512(<16 x float>, <16 x i32>)
declare <16 x i32>   @llvm.x86.avx512.permvar.si.512(<16 x i32>, <16 x i32>)
declare <8 x double> @llvm.x86.avx512.permvar.df.512(<8 x double>, <8 x i64>)
declare <8 x i64>    @llvm.x86.avx512.permvar.di.512(<8 x i64>, <8 x i64>)

declare <16 x i32>   @llvm.x86.avx512.mask.conflict.d.512(<16 x i32>, <16 x i32>, i16)
declare <8 x i64>    @llvm.x86.avx512.mask.conflict.q.512(<8 x i64>, <8 x i64>, i8)

declare <16 x i32>   @llvm.ctlz.v16i32(<16 x i32>, i1)
declare <8 x i64>    @llvm.ctlz.v8i64(<8 x i64>, i1)

attributes #1 = { alwaysinline norecurse nounwind "target-features"="+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+avx512f,+avx512vl,+avx512dq,+avx512cd" }
attributes #2 = { alwaysinline norecurse nounwind "target-features"="+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3" }
