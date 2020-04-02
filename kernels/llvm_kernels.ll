define internal void @ek_scatter_add_v1f32(i8* nocapture, i32, i32) local_unnamed_addr alwaysinline #1 {
  %r4 = bitcast i32 %1 to float
  %r5 = bitcast i8* %0 to float*
  %r6 = sext i32 %2 to i64
  %r7 = getelementptr inbounds float, float* %r5, i64 %r6
  %r8 = load float, float* %r7, align 4
  %r9 = fadd float %r8, %r4
  store float %r9, float* %r7, align 4
  ret void
}

define internal void @ek_masked_scatter_add_v1f32(i8* nocapture, i32, i32, i1 zeroext) local_unnamed_addr alwaysinline #1 {
  br i1 %3, label %L5, label %L12

L5:
  %r6 = bitcast i32 %1 to float
  %r7 = bitcast i8* %0 to float*
  %r8 = sext i32 %2 to i64
  %r9 = getelementptr inbounds float, float* %r7, i64 %r8
  %r10 = load float, float* %r9, align 4
  %r11 = fadd float %r10, %r6
  store float %r11, float* %r9, align 4
  br label %L12

L12:
  ret void
}

define internal void @ek_scatter_add_v1i32(i8* nocapture, i32, i32) local_unnamed_addr alwaysinline #1 {
  %r4 = bitcast i8* %0 to i32*
  %r5 = sext i32 %2 to i64
  %r6 = getelementptr inbounds i32, i32* %r4, i64 %r5
  %r7 = load i32, i32* %r6, align 4
  %r8 = add nsw i32 %r7, %1
  store i32 %r8, i32* %r6, align 4
  ret void
}

define internal void @ek_masked_scatter_add_v1i32(i8* nocapture, i32, i32, i1 zeroext) local_unnamed_addr alwaysinline #1 {
  br i1 %3, label %L5, label %L11

L5:
  %r6 = bitcast i8* %0 to i32*
  %r7 = sext i32 %2 to i64
  %r8 = getelementptr inbounds i32, i32* %r6, i64 %r7
  %r9 = load i32, i32* %r8, align 4
  %r10 = add nsw i32 %r9, %1
  store i32 %r10, i32* %r8, align 4
  br label %L11

L11:
  ret void
}

define internal void @ek_scatter_add_v1f64(i8* nocapture, double, i32) local_unnamed_addr alwaysinline #1 {
  %r4 = bitcast double %1 to <2 x float>
  %r5 = extractelement <2 x float> %r4, i32 0
  %r6 = fpext float %r5 to double
  %r7 = bitcast i8* %0 to double*
  %r8 = sext i32 %2 to i64
  %r9 = getelementptr inbounds double, double* %r7, i64 %r8
  %r10 = load double, double* %r9, align 8
  %r11 = fadd double %r10, %r6
  store double %r11, double* %r9, align 8
  ret void
}

define internal void @ek_masked_scatter_add_v1f64(i8* nocapture, double, i32, i1 zeroext) local_unnamed_addr alwaysinline #1 {
  br i1 %3, label %L5, label %L14

L5:
  %r6 = bitcast double %1 to <2 x float>
  %r7 = extractelement <2 x float> %r6, i32 0
  %r8 = fpext float %r7 to double
  %r9 = bitcast i8* %0 to double*
  %r10 = sext i32 %2 to i64
  %r11 = getelementptr inbounds double, double* %r9, i64 %r10
  %r12 = load double, double* %r11, align 8
  %r13 = fadd double %r12, %r8
  store double %r13, double* %r11, align 8
  br label %L14

L14:
  ret void
}

define internal void @ek_scatter_add_v1i64(i8* nocapture, double, i32) local_unnamed_addr alwaysinline #1 {
  %r4 = bitcast double %1 to i64
  %r5 = bitcast i8* %0 to i64*
  %r6 = sext i32 %2 to i64
  %r7 = getelementptr inbounds i64, i64* %r5, i64 %r6
  %r8 = load i64, i64* %r7, align 8
  %r9 = add nsw i64 %r8, %r4
  store i64 %r9, i64* %r7, align 8
  ret void
}

define internal void @ek_masked_scatter_add_v1i64(i8* nocapture, double, i32, i1 zeroext) local_unnamed_addr alwaysinline #1 {
  br i1 %3, label %L5, label %L12

L5:
  %r6 = bitcast double %1 to i64
  %r7 = bitcast i8* %0 to i64*
  %r8 = sext i32 %2 to i64
  %r9 = getelementptr inbounds i64, i64* %r7, i64 %r8
  %r10 = load i64, i64* %r9, align 8
  %r11 = add nsw i64 %r10, %r6
  store i64 %r11, i64* %r9, align 8
  br label %L12

L12:
  ret void
}

define internal void @ek_scatter_add_v16f32(i8*, <16 x float>, <16 x i32>) local_unnamed_addr alwaysinline #1 {
L3:
  %r4 = tail call <16 x float> @llvm.x86.avx512.mask.gather.dps.512(<16 x float> undef, i8* %0, <16 x i32> %2, <16 x i1> <i1 1, i1 1, i1 1, i1 1, i1 1, i1 1, i1 1, i1 1, i1 1, i1 1, i1 1, i1 1, i1 1, i1 1, i1 1, i1 1>, i32 4)
  %r5 = tail call <16 x i32> @llvm.x86.avx512.conflict.d.512(<16 x i32> %2) #3
  %r6 = icmp ne <16 x i32> %r5, zeroinitializer
  %r7 = bitcast <16 x i1> %r6 to i16
  %r8 = icmp eq i16 %r7, 0
  br i1 %r8, label %L24, label %L9, !prof !{!"branch_weights", i32 2000, i32 1}

L9:
  %r10 = tail call <16 x i32> @llvm.ctlz.v16i32(<16 x i32> %r5, i1 false) #3
  %r11 = sub nsw <16 x i32> <i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31>, %r10
  br label %L12

L12:
  %r13 = phi <16 x i32> [ %r11, %L9 ], [ %r19, %L12 ]
  %r14 = phi <16 x i1> [ %r6, %L9 ], [ %r21, %L12 ]
  %r15 = phi <16 x float> [ %1, %L9 ], [ %r20, %L12 ]
  %r16 = tail call <16 x float> @llvm.x86.avx512.permvar.sf.512(<16 x float> %r15, <16 x i32> %r13) #3
  %r17 = select <16 x i1> %r14, <16 x float> %r16, <16 x float> zeroinitializer
  %r18 = tail call <16 x i32> @llvm.x86.avx512.permvar.si.512(<16 x i32> %r13, <16 x i32> %r13) #3
  %r19 = select <16 x i1> %r14, <16 x i32> %r18, <16 x i32> %r13
  %r20 = fadd <16 x float> %r15, %r17
  %r21 = icmp ne <16 x i32> %r19, <i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1>
  %r22 = bitcast <16 x i1> %r21 to i16
  %r23 = icmp eq i16 %r22, 0
  br i1 %r23, label %L24, label %L12

L24:
  %r25 = phi <16 x float> [ %1, %L3 ], [ %r20, %L12 ]
  %r26 = fadd <16 x float> %r4, %r25
  tail call void @llvm.x86.avx512.mask.scatter.dps.512(i8* %0, <16 x i1> <i1 1, i1 1, i1 1, i1 1, i1 1, i1 1, i1 1, i1 1, i1 1, i1 1, i1 1, i1 1, i1 1, i1 1, i1 1, i1 1>, <16 x i32> %2, <16 x float> %r26, i32 4)
  ret void
}

define internal void @ek_masked_scatter_add_v16f32(i8*, <16 x float>, <16 x i32>, i16 zeroext) local_unnamed_addr alwaysinline #1 {
L4:
  %r5 = bitcast i16 %3 to <16 x i1>
  %r6 = tail call <16 x float> @llvm.x86.avx512.mask.gather.dps.512(<16 x float> undef, i8* %0, <16 x i32> %2, <16 x i1> %r5, i32 4)
  %r7 = tail call <16 x i32> @llvm.x86.avx512.conflict.d.512(<16 x i32> %2) #3
  %r8 = zext i16 %3 to i32
  %r9 = insertelement <16 x i32> undef, i32 %r8, i32 0
  %r10 = shufflevector <16 x i32> %r9, <16 x i32> undef, <16 x i32> zeroinitializer
  %r11 = and <16 x i32> %r7, %r10
  %r12 = icmp ne <16 x i32> %r11, zeroinitializer
  %r13 = bitcast <16 x i1> %r12 to i16
  %r14 = icmp eq i16 %r13, 0
  br i1 %r14, label %L31, label %L15, !prof !{!"branch_weights", i32 2000, i32 1}

L15:
  %r16 = tail call <16 x i32> @llvm.ctlz.v16i32(<16 x i32> %r11, i1 false) #3
  %r17 = sub nsw <16 x i32> <i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31>, %r16
  br label %L18

L18:
  %r19 = phi <16 x i32> [ %r17, %L15 ], [ %r25, %L18 ]
  %r20 = phi <16 x i1> [ %r12, %L15 ], [ %r28, %L18 ]
  %r21 = phi <16 x float> [ %1, %L15 ], [ %r26, %L18 ]
  %r22 = tail call <16 x float> @llvm.x86.avx512.permvar.sf.512(<16 x float> %r21, <16 x i32> %r19) #3
  %r23 = select <16 x i1> %r20, <16 x float> %r22, <16 x float> zeroinitializer
  %r24 = tail call <16 x i32> @llvm.x86.avx512.permvar.si.512(<16 x i32> %r19, <16 x i32> %r19) #3
  %r25 = select <16 x i1> %r20, <16 x i32> %r24, <16 x i32> %r19
  %r26 = fadd <16 x float> %r21, %r23
  %r27 = icmp ne <16 x i32> %r25, <i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1>
  %r28 = and <16 x i1> %r27, %r5
  %r29 = bitcast <16 x i1> %r28 to i16
  %r30 = icmp eq i16 %r29, 0
  br i1 %r30, label %L31, label %L18

L31:
  %r32 = phi <16 x float> [ %1, %L4 ], [ %r26, %L18 ]
  %r33 = fadd <16 x float> %r6, %r32
  tail call void @llvm.x86.avx512.mask.scatter.dps.512(i8* %0, <16 x i1> %r5, <16 x i32> %2, <16 x float> %r33, i32 4)
  ret void
}

define internal void @ek_scatter_add_v16i32(i8*, <16 x i32>, <16 x i32>) local_unnamed_addr alwaysinline #1 {
L3:
  %r4 = tail call <16 x i32> @llvm.x86.avx512.mask.gather.dpi.512(<16 x i32> undef, i8* %0, <16 x i32> %2, <16 x i1> <i1 1, i1 1, i1 1, i1 1, i1 1, i1 1, i1 1, i1 1, i1 1, i1 1, i1 1, i1 1, i1 1, i1 1, i1 1, i1 1>, i32 4)
  %r5 = tail call <16 x i32> @llvm.x86.avx512.conflict.d.512(<16 x i32> %2) #3
  %r6 = icmp ne <16 x i32> %r5, zeroinitializer
  %r7 = bitcast <16 x i1> %r6 to i16
  %r8 = icmp eq i16 %r7, 0
  br i1 %r8, label %L24, label %L9, !prof !{!"branch_weights", i32 2000, i32 1}

L9:
  %r10 = tail call <16 x i32> @llvm.ctlz.v16i32(<16 x i32> %r5, i1 false) #3
  %r11 = sub nsw <16 x i32> <i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31>, %r10
  br label %L12

L12:
  %r13 = phi <16 x i32> [ %r11, %L9 ], [ %r19, %L12 ]
  %r14 = phi <16 x i1> [ %r6, %L9 ], [ %r21, %L12 ]
  %r15 = phi <16 x i32> [ %1, %L9 ], [ %r20, %L12 ]
  %r16 = tail call <16 x i32> @llvm.x86.avx512.permvar.si.512(<16 x i32> %r15, <16 x i32> %r13) #3
  %r17 = select <16 x i1> %r14, <16 x i32> %r16, <16 x i32> zeroinitializer
  %r18 = tail call <16 x i32> @llvm.x86.avx512.permvar.si.512(<16 x i32> %r13, <16 x i32> %r13) #3
  %r19 = select <16 x i1> %r14, <16 x i32> %r18, <16 x i32> %r13
  %r20 = add <16 x i32> %r17, %r15
  %r21 = icmp ne <16 x i32> %r19, <i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1>
  %r22 = bitcast <16 x i1> %r21 to i16
  %r23 = icmp eq i16 %r22, 0
  br i1 %r23, label %L24, label %L12

L24:
  %r25 = phi <16 x i32> [ %1, %L3 ], [ %r20, %L12 ]
  %r26 = add <16 x i32> %r25, %r4
  tail call void @llvm.x86.avx512.mask.scatter.dpi.512(i8* %0, <16 x i1> <i1 1, i1 1, i1 1, i1 1, i1 1, i1 1, i1 1, i1 1, i1 1, i1 1, i1 1, i1 1, i1 1, i1 1, i1 1, i1 1>, <16 x i32> %2, <16 x i32> %r26, i32 4)
  ret void
}

define internal void @ek_masked_scatter_add_v16i32(i8*, <16 x i32>, <16 x i32>, i16 zeroext) local_unnamed_addr alwaysinline #1 {
L4:
  %r5 = bitcast i16 %3 to <16 x i1>
  %r6 = tail call <16 x i32> @llvm.x86.avx512.mask.gather.dpi.512(<16 x i32> undef, i8* %0, <16 x i32> %2, <16 x i1> %r5, i32 4)
  %r7 = tail call <16 x i32> @llvm.x86.avx512.conflict.d.512(<16 x i32> %2) #3
  %r8 = zext i16 %3 to i32
  %r9 = insertelement <16 x i32> undef, i32 %r8, i32 0
  %r10 = shufflevector <16 x i32> %r9, <16 x i32> undef, <16 x i32> zeroinitializer
  %r11 = and <16 x i32> %r7, %r10
  %r12 = icmp ne <16 x i32> %r11, zeroinitializer
  %r13 = bitcast <16 x i1> %r12 to i16
  %r14 = icmp eq i16 %r13, 0
  br i1 %r14, label %L31, label %L15, !prof !{!"branch_weights", i32 2000, i32 1}

L15:
  %r16 = tail call <16 x i32> @llvm.ctlz.v16i32(<16 x i32> %r11, i1 false) #3
  %r17 = sub nsw <16 x i32> <i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31>, %r16
  br label %L18

L18:
  %r19 = phi <16 x i32> [ %r17, %L15 ], [ %r25, %L18 ]
  %r20 = phi <16 x i1> [ %r12, %L15 ], [ %r28, %L18 ]
  %r21 = phi <16 x i32> [ %1, %L15 ], [ %r26, %L18 ]
  %r22 = tail call <16 x i32> @llvm.x86.avx512.permvar.si.512(<16 x i32> %r21, <16 x i32> %r19) #3
  %r23 = select <16 x i1> %r20, <16 x i32> %r22, <16 x i32> zeroinitializer
  %r24 = tail call <16 x i32> @llvm.x86.avx512.permvar.si.512(<16 x i32> %r19, <16 x i32> %r19) #3
  %r25 = select <16 x i1> %r20, <16 x i32> %r24, <16 x i32> %r19
  %r26 = add <16 x i32> %r23, %r21
  %r27 = icmp ne <16 x i32> %r25, <i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1>
  %r28 = and <16 x i1> %r27, %r5
  %r29 = bitcast <16 x i1> %r28 to i16
  %r30 = icmp eq i16 %r29, 0
  br i1 %r30, label %L31, label %L18

L31:
  %r32 = phi <16 x i32> [ %1, %L4 ], [ %r26, %L18 ]
  %r33 = add <16 x i32> %r32, %r6
  tail call void @llvm.x86.avx512.mask.scatter.dpi.512(i8* %0, <16 x i1> %r5, <16 x i32> %2, <16 x i32> %r33, i32 4)
  ret void
}

define internal void @ek_scatter_add_v8f64(i8*, <16 x float>, <8 x i64>) local_unnamed_addr alwaysinline #1 {
L3:
  %r4 = tail call <8 x double> @llvm.x86.avx512.mask.gather.qpd.512(<8 x double> undef, i8* %0, <8 x i64> %2, <8 x i1> <i1 1, i1 1, i1 1, i1 1, i1 1, i1 1, i1 1, i1 1>, i32 8)
  %r5 = tail call <8 x i64> @llvm.x86.avx512.conflict.q.512(<8 x i64> %2) #3
  %r6 = icmp ne <8 x i64> %r5, zeroinitializer
  %r7 = bitcast <8 x i1> %r6 to i8
  %r8 = icmp eq i8 %r7, 0
  br i1 %r8, label %L26, label %L9, !prof !{!"branch_weights", i32 2000, i32 1}

L9:
  %r10 = tail call <8 x i64> @llvm.ctlz.v8i64(<8 x i64> %r5, i1 false) #3
  %r11 = sub nsw <8 x i64> <i64 63, i64 63, i64 63, i64 63, i64 63, i64 63, i64 63, i64 63>, %r10
  br label %L12

L12:
  %r13 = phi <8 x i64> [ %r11, %L9 ], [ %r20, %L12 ]
  %r14 = phi <8 x i1> [ %r6, %L9 ], [ %r23, %L12 ]
  %r15 = phi <16 x float> [ %1, %L9 ], [ %r22, %L12 ]
  %r16 = bitcast <16 x float> %r15 to <8 x double>
  %r17 = tail call <8 x double> @llvm.x86.avx512.permvar.df.512(<8 x double> %r16, <8 x i64> %r13) #3
  %r18 = select <8 x i1> %r14, <8 x double> %r17, <8 x double> zeroinitializer
  %r19 = tail call <8 x i64> @llvm.x86.avx512.permvar.di.512(<8 x i64> %r13, <8 x i64> %r13) #3
  %r20 = select <8 x i1> %r14, <8 x i64> %r19, <8 x i64> %r13
  %r21 = fadd <8 x double> %r18, %r16
  %r22 = bitcast <8 x double> %r21 to <16 x float>
  %r23 = icmp ne <8 x i64> %r20, <i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1>
  %r24 = bitcast <8 x i1> %r23 to i8
  %r25 = icmp eq i8 %r24, 0
  br i1 %r25, label %L26, label %L12

L26:
  %r27 = phi <16 x float> [ %1, %L3 ], [ %r22, %L12 ]
  %r28 = bitcast <16 x float> %r27 to <8 x double>
  %r29 = fadd <8 x double> %r4, %r28
  tail call void @llvm.x86.avx512.mask.scatter.qpd.512(i8* %0, <8 x i1> <i1 1, i1 1, i1 1, i1 1, i1 1, i1 1, i1 1, i1 1>, <8 x i64> %2, <8 x double> %r29, i32 8)
  ret void
}

define internal void @ek_masked_scatter_add_v8f64(i8*, <16 x float>, <8 x i64>, i8 zeroext) local_unnamed_addr alwaysinline #1 {
L4:
  %r5 = bitcast i8 %3 to <8 x i1>
  %r6 = tail call <8 x double> @llvm.x86.avx512.mask.gather.qpd.512(<8 x double> undef, i8* %0, <8 x i64> %2, <8 x i1> %r5, i32 8)
  %r7 = tail call <8 x i64> @llvm.x86.avx512.conflict.q.512(<8 x i64> %2) #3
  %r8 = zext i8 %3 to i64
  %r9 = insertelement <8 x i64> undef, i64 %r8, i32 0
  %r10 = shufflevector <8 x i64> %r9, <8 x i64> undef, <8 x i32> zeroinitializer
  %r11 = and <8 x i64> %r7, %r10
  %r12 = icmp ne <8 x i64> %r11, zeroinitializer
  %r13 = bitcast <8 x i1> %r12 to i8
  %r14 = icmp eq i8 %r13, 0
  br i1 %r14, label %L33, label %L15, !prof !{!"branch_weights", i32 2000, i32 1}

L15:
  %r16 = tail call <8 x i64> @llvm.ctlz.v8i64(<8 x i64> %r11, i1 false) #3
  %r17 = sub nsw <8 x i64> <i64 63, i64 63, i64 63, i64 63, i64 63, i64 63, i64 63, i64 63>, %r16
  br label %L18

L18:
  %r19 = phi <8 x i64> [ %r17, %L15 ], [ %r26, %L18 ]
  %r20 = phi <8 x i1> [ %r12, %L15 ], [ %r30, %L18 ]
  %r21 = phi <16 x float> [ %1, %L15 ], [ %r28, %L18 ]
  %r22 = bitcast <16 x float> %r21 to <8 x double>
  %r23 = tail call <8 x double> @llvm.x86.avx512.permvar.df.512(<8 x double> %r22, <8 x i64> %r19) #3
  %r24 = select <8 x i1> %r20, <8 x double> %r23, <8 x double> zeroinitializer
  %r25 = tail call <8 x i64> @llvm.x86.avx512.permvar.di.512(<8 x i64> %r19, <8 x i64> %r19) #3
  %r26 = select <8 x i1> %r20, <8 x i64> %r25, <8 x i64> %r19
  %r27 = fadd <8 x double> %r24, %r22
  %r28 = bitcast <8 x double> %r27 to <16 x float>
  %r29 = icmp ne <8 x i64> %r26, <i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1>
  %r30 = and <8 x i1> %r29, %r5
  %r31 = bitcast <8 x i1> %r30 to i8
  %r32 = icmp eq i8 %r31, 0
  br i1 %r32, label %L33, label %L18

L33:
  %r34 = phi <16 x float> [ %1, %L4 ], [ %r28, %L18 ]
  %r35 = bitcast <16 x float> %r34 to <8 x i64>
  %r36 = bitcast <8 x double> %r6 to <8 x i64>
  %r37 = add <8 x i64> %r35, %r36
  tail call void @llvm.x86.avx512.mask.scatter.qpq.512(i8* %0, <8 x i1> %r5, <8 x i64> %2, <8 x i64> %r37, i32 8)
  ret void
}

define internal void @ek_scatter_add_v8i64(i8*, <8 x i64>, <8 x i64>) local_unnamed_addr alwaysinline #1 {
L3:
  %r4 = tail call <8 x i64> @llvm.x86.avx512.mask.gather.qpq.512(<8 x i64> undef, i8* %0, <8 x i64> %2, <8 x i1> <i1 1, i1 1, i1 1, i1 1, i1 1, i1 1, i1 1, i1 1>, i32 8)
  %r5 = tail call <8 x i64> @llvm.x86.avx512.conflict.q.512(<8 x i64> %2) #3
  %r6 = icmp ne <8 x i64> %r5, zeroinitializer
  %r7 = bitcast <8 x i1> %r6 to i8
  %r8 = icmp eq i8 %r7, 0
  br i1 %r8, label %L24, label %L9, !prof !{!"branch_weights", i32 2000, i32 1}

L9:
  %r10 = tail call <8 x i64> @llvm.ctlz.v8i64(<8 x i64> %r5, i1 false) #3
  %r11 = sub nsw <8 x i64> <i64 63, i64 63, i64 63, i64 63, i64 63, i64 63, i64 63, i64 63>, %r10
  br label %L12

L12:
  %r13 = phi <8 x i64> [ %r11, %L9 ], [ %r19, %L12 ]
  %r14 = phi <8 x i1> [ %r6, %L9 ], [ %r21, %L12 ]
  %r15 = phi <8 x i64> [ %1, %L9 ], [ %r20, %L12 ]
  %r16 = tail call <8 x i64> @llvm.x86.avx512.permvar.di.512(<8 x i64> %r15, <8 x i64> %r13) #3
  %r17 = select <8 x i1> %r14, <8 x i64> %r16, <8 x i64> zeroinitializer
  %r18 = tail call <8 x i64> @llvm.x86.avx512.permvar.di.512(<8 x i64> %r13, <8 x i64> %r13) #3
  %r19 = select <8 x i1> %r14, <8 x i64> %r18, <8 x i64> %r13
  %r20 = add <8 x i64> %r17, %r15
  %r21 = icmp ne <8 x i64> %r19, <i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1>
  %r22 = bitcast <8 x i1> %r21 to i8
  %r23 = icmp eq i8 %r22, 0
  br i1 %r23, label %L24, label %L12

L24:
  %r25 = phi <8 x i64> [ %1, %L3 ], [ %r20, %L12 ]
  %r26 = add <8 x i64> %r25, %r4
  tail call void @llvm.x86.avx512.mask.scatter.qpq.512(i8* %0, <8 x i1> <i1 1, i1 1, i1 1, i1 1, i1 1, i1 1, i1 1, i1 1>, <8 x i64> %2, <8 x i64> %r26, i32 8)
  ret void
}

define internal void @ek_masked_scatter_add_v8i64(i8*, <8 x i64>, <8 x i64>, i8 zeroext) local_unnamed_addr alwaysinline #1 {
L4:
  %r5 = bitcast i8 %3 to <8 x i1>
  %r6 = tail call <8 x i64> @llvm.x86.avx512.mask.gather.qpq.512(<8 x i64> undef, i8* %0, <8 x i64> %2, <8 x i1> %r5, i32 8)
  %r7 = tail call <8 x i64> @llvm.x86.avx512.conflict.q.512(<8 x i64> %2) #3
  %r8 = zext i8 %3 to i64
  %r9 = insertelement <8 x i64> undef, i64 %r8, i32 0
  %r10 = shufflevector <8 x i64> %r9, <8 x i64> undef, <8 x i32> zeroinitializer
  %r11 = and <8 x i64> %r7, %r10
  %r12 = icmp ne <8 x i64> %r11, zeroinitializer
  %r13 = bitcast <8 x i1> %r12 to i8
  %r14 = icmp eq i8 %r13, 0
  br i1 %r14, label %L31, label %L15, !prof !{!"branch_weights", i32 2000, i32 1}

L15:
  %r16 = tail call <8 x i64> @llvm.ctlz.v8i64(<8 x i64> %r11, i1 false) #3
  %r17 = sub nsw <8 x i64> <i64 63, i64 63, i64 63, i64 63, i64 63, i64 63, i64 63, i64 63>, %r16
  br label %L18

L18:
  %r19 = phi <8 x i64> [ %r17, %L15 ], [ %r25, %L18 ]
  %r20 = phi <8 x i1> [ %r12, %L15 ], [ %r28, %L18 ]
  %r21 = phi <8 x i64> [ %1, %L15 ], [ %r26, %L18 ]
  %r22 = tail call <8 x i64> @llvm.x86.avx512.permvar.di.512(<8 x i64> %r21, <8 x i64> %r19) #3
  %r23 = select <8 x i1> %r20, <8 x i64> %r22, <8 x i64> zeroinitializer
  %r24 = tail call <8 x i64> @llvm.x86.avx512.permvar.di.512(<8 x i64> %r19, <8 x i64> %r19) #3
  %r25 = select <8 x i1> %r20, <8 x i64> %r24, <8 x i64> %r19
  %r26 = add <8 x i64> %r23, %r21
  %r27 = icmp ne <8 x i64> %r25, <i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1>
  %r28 = and <8 x i1> %r27, %r5
  %r29 = bitcast <8 x i1> %r28 to i8
  %r30 = icmp eq i8 %r29, 0
  br i1 %r30, label %L31, label %L18

L31:
  %r32 = phi <8 x i64> [ %1, %L4 ], [ %r26, %L18 ]
  %r33 = add <8 x i64> %r32, %r6
  tail call void @llvm.x86.avx512.mask.scatter.qpq.512(i8* %0, <8 x i1> %r5, <8 x i64> %2, <8 x i64> %r33, i32 8)
  ret void
}

declare <16 x i32>   @llvm.ctlz.v16i32(<16 x i32>, i1) #4
declare <8 x i64>    @llvm.ctlz.v8i64(<8 x i64>, i1) #4

declare <16 x i32>   @llvm.x86.avx512.conflict.d.512(<16 x i32>) #4
declare <8 x i64>    @llvm.x86.avx512.conflict.q.512(<8 x i64>) #4

declare <16 x i32>   @llvm.x86.avx512.permvar.si.512(<16 x i32>, <16 x i32>) #4
declare <16 x float> @llvm.x86.avx512.permvar.sf.512(<16 x float>, <16 x i32>) #4
declare <8 x i64>    @llvm.x86.avx512.permvar.di.512(<8 x i64>, <8 x i64>) #4
declare <8 x double> @llvm.x86.avx512.permvar.df.512(<8 x double>, <8 x i64>) #4

declare <16 x i32>   @llvm.x86.avx512.mask.gather.dpi.512(<16 x i32>, i8*, <16 x i32>, <16 x i1>, i32) #2
declare <16 x float> @llvm.x86.avx512.mask.gather.dps.512(<16 x float>, i8*, <16 x i32>, <16 x i1>, i32) #2
declare <8 x i64>    @llvm.x86.avx512.mask.gather.qpq.512(<8 x i64>, i8*, <8 x i64>, <8 x i1>, i32) #2
declare <8 x double> @llvm.x86.avx512.mask.gather.qpd.512(<8 x double>, i8*, <8 x i64>, <8 x i1>, i32) #2

declare void @llvm.x86.avx512.mask.scatter.dpi.512(i8*, <16 x i1>, <16 x i32>, <16 x i32>, i32) #3
declare void @llvm.x86.avx512.mask.scatter.dps.512(i8*, <16 x i1>, <16 x i32>, <16 x float>, i32) #3
declare void @llvm.x86.avx512.mask.scatter.qpq.512(i8*, <8 x i1>, <8 x i64>, <8 x i64>, i32) #3
declare void @llvm.x86.avx512.mask.scatter.qpd.512(i8*, <8 x i1>, <8 x i64>, <8 x double>, i32) #3

attributes #1 = { alwaysinline norecurse nounwind "no-frame-pointer-elim"="false" "target-features"="+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+avx512f,+avx512vl,+avx512dq,+avx512cd" }
attributes #2 = { nounwind readonly }
attributes #3 = { nounwind }
attributes #4 = { nounwind readnone }

