# This (very hacky) script was used to convert Clang LLVM IR output for
# 'scatter_add.c' into extra-compatible IR that works with LLVM 7/8/9/10,
# and which has the right calling conventions so that we can invoke it
# from Enoki kernels. It is very likely never needed again but checked
# in here just in case..

import re

x = open("scatter_add.ll", "r").read()

# Comments
x = re.sub(r'^;.*\n', '', x, flags=re.MULTILINE)
x = re.sub(r' *;.*\n', '\n', x)

# Registers
x = re.sub(r'%', '%r', x)

# Labels
x = re.sub(r'label %r', 'label %L', x)
x = re.sub(r' %r([0-9]*) \]', r' %L\1 ]', x)
x = re.sub(r'^([0-9]*):', r'L\1:', x, flags=re.MULTILINE)

# TBAA
x = re.sub(r', !tbaa ![0-9]*', '', x)

# Misc
x = re.sub(r'i1 true', 'i1 1', x)
x = re.sub(r', !misexpect ![0-9]*', '', x)
x = re.sub(r'!prof !12', r'!prof !{!"branch_weights", i32 2000, i32 1}', x)

# Scalar
x = re.sub(r'extractelement <4 x ([a-z0-9]*)> %r([0-9]), i32 3',
           r'extractelement <1 x \1> %r\2, i32 0',
           x)

x = re.sub(r'<4 x ([a-z0-9]*)>', r'<1 x \1>', x)
x = re.sub('i1 zeroext %r3', '<1 x i1> %r3', x)

x = re.sub(r'(br i1 %r3, label %L[0-9]*, label %L[0-9]*)',
           r'%r3 = extractelement <1 x i1> %3, i32 0\n  \1', x)

x = re.sub(r'i16 %r3 ', 'i16 %3 ', x)
x = re.sub(r'i8 %r3 ', 'i8 %3 ', x)
x = re.sub(r'i8 zeroext', '<8 x i1>', x)
x = re.sub(r'i16 zeroext', '<16 x i1>', x)

x = re.sub(r'%r5 = bitcast i16 %3 to <16 x i1>',
           r'%r5 = bitcast <16 x i1> %3 to i16', x)

x = re.sub(r'%r5 = bitcast i8 %3 to <8 x i1>',
           r'%r5 = bitcast <8 x i1> %3 to i8', x)

x = re.sub(r'%r8 = zext i16 %3 to i32',
           r'%r8 = zext i16 %r5 to i32', x)

x = re.sub(r'%r8 = zext i8 %3 to i64',
           r'%r8 = zext i8 %r5 to i64', x)

x = re.sub(r'<8 x i1> %r5', '<8 x i1> %3', x)
x = re.sub(r'<16 x i1> %r5', '<16 x i1> %3', x)

x = re.sub(r'and <16 x i1> %r27, %r5', 'and <16 x i1> %r27, %3', x)
x = re.sub(r'and <8 x i1> %r27, %r5', 'and <8 x i1> %r27, %3', x)

# Function annotations
x = re.sub(r'hidden', r'internal', x)

x = re.sub(r'ek_', r'ek.', x)
x = re.sub(' immarg', '', x)
x = re.sub('#0', '#1', x)
x = re.sub('#1', '#1', x)
x = re.sub('#2', '#1', x)
x = re.sub(' #[3-9]', '', x)

# Function parameters
x = re.sub('%r0(?=[^0-9])', '%0', x)
x = re.sub('%r1(?=[^0-9])', '%1', x)
x = re.sub('%r2(?=[^0-9])', '%2', x)

out = ''
suffix = ''
for line in x.split('\n'):
    if line.startswith('!') or line.startswith('attributes') or \
       line.startswith('source') or line.startswith('target'):
        continue
    if 'gather' in line:
        line = line.replace('zeroinitializer', 'undef')
    if line.startswith('define'):
        if '_v1f' in line or '_v1i' in line:
            line = re.sub('#1', '#2', line)
        params = re.findall(' %r?([0-9]*)', line)
        line = re.sub(' %r?[0-9]*', '', line)
        line += '\nL%i:' % (max([int(i) for i in params]) + 1)
        pass
    if 'declare' in line:
        suffix += line + '\n'
    else:
        out += line + '\n'

out += suffix
out = re.sub(r'\n\n+', r'\n\n', out)
out = re.sub(' *$', '', out)
out = out[1:]
out += '\nattributes #1 = { alwaysinline norecurse nounwind "frame-pointer"="none" "target-features"="+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+avx512f,+avx512vl,+avx512dq,+avx512cd" }'
out += '\nattributes #2 = { alwaysinline norecurse nounwind "frame-pointer"="none" "target-features"="+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3" }'
print(out)

# add top labels, undef in gathers
