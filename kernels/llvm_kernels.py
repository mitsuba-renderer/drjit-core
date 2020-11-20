# This (very hacky) script was used to convert Clang LLVM IR output for
# 'scatter_add.c' into extra-compatible IR that works with LLVM 7/8/9/10,
# and which has the right calling conventions so that we can invoke it
# from Enoki kernels. It is very likely never needed again but checked
# in here just in case..

import re

x = open("scatter_add.ll", "r").read()

# Registers
x = re.sub(r'%', '%r', x)

# Labels:
for i1 ,i2 in { 5: 0, 13: 1, 16: 1, 17: 2, 21: 2, 23: 3, 27: 3}.items():
    x = x.replace('; <label>:%i:' % i1, 'L%i:;' % i2)
    x = x.replace('label %%r%i' % i1, 'label %%L%i' % i2)
    x = x.replace('%%r%i ]' % i1, '%%L%i ]' % i2)

# Comments
x = re.sub(r'^;.*\n', '', x, flags=re.MULTILINE)
x = re.sub(r' *;.*\n', '\n', x)

# Prelude
x = re.sub(r'^source.*\n', '', x, flags=re.MULTILINE)
x = re.sub(r'^target.*\n', '', x, flags=re.MULTILINE)
x = re.sub(r'^!.*\n', '', x, flags=re.MULTILINE)
x = re.sub(r'^attributes .*\n', '', x, flags=re.MULTILINE)

# Function parameters
x = re.sub('%r0(?=[^0-9])', '%0', x)
x = re.sub('%r1(?=[^0-9])', '%1', x)
x = re.sub('%r2(?=[^0-9])', '%2', x)

# Fake vectors
x = re.sub(r'<4 x ([a-z0-9]*)>', r'<1 x \1>', x)
x = x.replace('i1 zeroext', '<1 x i1>')

# Misc
x = x.replace(' nocapture', '')
x = x.replace('br i1 %r3', '%r3 = extractelement <1 x i1> %3, i32 0\n  br i1 %r3')
x = x.replace('#1', '#2')
x = x.replace('#0', '#2')
x = x.replace('@scat', '@ek.scat')

x += '\nattributes #2 = { alwaysinline norecurse nounwind "frame-pointer"="none" }'
x = re.sub('\n\n+', '\n\n', x)
x = x[1:]
print(x)
