define void @ek.scatter_add_f32(i8*, <1 x float>, <1 x i32>, <1 x i1>) local_unnamed_addr #2 {
  %r3 = extractelement <1 x i1> %3, i32 0
  br i1 %r3, label %L0, label %L3

L0:
  %r6 = bitcast i8* %0 to i32*
  %r7 = extractelement <1 x i32> %2, i32 0
  %r8 = zext i32 %r7 to i64
  %r9 = getelementptr inbounds i32, i32* %r6, i64 %r8
  %r10 = load atomic i32, i32* %r9 acquire, align 4
  %r11 = bitcast i32 %r10 to float
  %r12 = extractelement <1 x float> %1, i32 0
  %r13 = fadd float %r12, %r11
  %r14 = bitcast float %r13 to i32
  %r15 = icmp eq i32 %r10, %r14
  br i1 %r15, label %L3, label %L1

L1:
  %r17 = phi i32 [ %r25, %L2 ], [ %r14, %L0 ]
  %r18 = phi i32 [ %r22, %L2 ], [ %r10, %L0 ]
  %r19 = cmpxchg weak i32* %r9, i32 %r18, i32 %r17 release monotonic
  %r20 = extractvalue { i32, i1 } %r19, 1
  br i1 %r20, label %L3, label %L2

L2:
  %r22 = extractvalue { i32, i1 } %r19, 0
  %r23 = bitcast i32 %r22 to float
  %r24 = fadd float %r12, %r23
  %r25 = bitcast float %r24 to i32
  %r26 = icmp eq i32 %r22, %r25
  br i1 %r26, label %L3, label %L1

L3:
  ret void
}

define void @ek.scatter_add_f64(i8*, <1 x double>, <1 x i32>, <1 x i1>) local_unnamed_addr #2 {
  %r3 = extractelement <1 x i1> %3, i32 0
  br i1 %r3, label %L0, label %L3

L0:
  %r6 = bitcast i8* %0 to i64*
  %r7 = extractelement <1 x i32> %2, i32 0
  %r8 = zext i32 %r7 to i64
  %r9 = getelementptr inbounds i64, i64* %r6, i64 %r8
  %r10 = load atomic i64, i64* %r9 acquire, align 8
  %r11 = bitcast i64 %r10 to double
  %r12 = extractelement <1 x double> %1, i32 0
  %r13 = fadd double %r12, %r11
  %r14 = bitcast double %r13 to i64
  %r15 = icmp eq i64 %r10, %r14
  br i1 %r15, label %L3, label %L1

L1:
  %r17 = phi i64 [ %r25, %L2 ], [ %r14, %L0 ]
  %r18 = phi i64 [ %r22, %L2 ], [ %r10, %L0 ]
  %r19 = cmpxchg weak i64* %r9, i64 %r18, i64 %r17 release monotonic
  %r20 = extractvalue { i64, i1 } %r19, 1
  br i1 %r20, label %L3, label %L2

L2:
  %r22 = extractvalue { i64, i1 } %r19, 0
  %r23 = bitcast i64 %r22 to double
  %r24 = fadd double %r12, %r23
  %r25 = bitcast double %r24 to i64
  %r26 = icmp eq i64 %r22, %r25
  br i1 %r26, label %L3, label %L1

L3:
  ret void
}

define void @ek.scatter_add_i32(i8*, <1 x i32>, <1 x i32>, <1 x i1>) local_unnamed_addr #2 {
  %r3 = extractelement <1 x i1> %3, i32 0
  br i1 %r3, label %L0, label %L3

L0:
  %r6 = bitcast i8* %0 to i32*
  %r7 = extractelement <1 x i32> %2, i32 0
  %r8 = zext i32 %r7 to i64
  %r9 = getelementptr inbounds i32, i32* %r6, i64 %r8
  %r10 = load atomic i32, i32* %r9 acquire, align 4
  %r11 = extractelement <1 x i32> %1, i32 0
  %r12 = icmp eq i32 %r11, 0
  br i1 %r12, label %L3, label %L1

L1:
  %r14 = add i32 %r10, %r11
  %r15 = cmpxchg weak i32* %r9, i32 %r10, i32 %r14 release monotonic
  %r16 = extractvalue { i32, i1 } %r15, 1
  br i1 %r16, label %L3, label %L2

L2:
  %r18 = phi { i32, i1 } [ %r21, %L2 ], [ %r15, %L1 ]
  %r19 = extractvalue { i32, i1 } %r18, 0
  %r20 = add i32 %r19, %r11
  %r21 = cmpxchg weak i32* %r9, i32 %r19, i32 %r20 release monotonic
  %r22 = extractvalue { i32, i1 } %r21, 1
  br i1 %r22, label %L3, label %L2

L3:
  ret void
}

define void @ek.scatter_add_i64(i8*, <1 x i64>, <1 x i32>, <1 x i1>) local_unnamed_addr #2 {
  %r3 = extractelement <1 x i1> %3, i32 0
  br i1 %r3, label %L0, label %L3

L0:
  %r6 = bitcast i8* %0 to i64*
  %r7 = extractelement <1 x i32> %2, i32 0
  %r8 = zext i32 %r7 to i64
  %r9 = getelementptr inbounds i64, i64* %r6, i64 %r8
  %r10 = load atomic i64, i64* %r9 acquire, align 8
  %r11 = extractelement <1 x i64> %1, i32 0
  %r12 = icmp eq i64 %r11, 0
  br i1 %r12, label %L3, label %L1

L1:
  %r14 = add i64 %r10, %r11
  %r15 = cmpxchg weak i64* %r9, i64 %r10, i64 %r14 release monotonic
  %r16 = extractvalue { i64, i1 } %r15, 1
  br i1 %r16, label %L3, label %L2

L2:
  %r18 = phi { i64, i1 } [ %r21, %L2 ], [ %r15, %L1 ]
  %r19 = extractvalue { i64, i1 } %r18, 0
  %r20 = add i64 %r19, %r11
  %r21 = cmpxchg weak i64* %r9, i64 %r19, i64 %r20 release monotonic
  %r22 = extractvalue { i64, i1 } %r21, 1
  br i1 %r22, label %L3, label %L2

L3:
  ret void
}

attributes #2 = { alwaysinline norecurse nounwind "frame-pointer"="none" }
