#xs = 0.01
#rs = 0.025
for scale in [1, 2, 4, 10, 20, 50, 100]:
    xs = 1. / scale
    rs = 1. / scale
    with open('51-2b-ss.CNG.swc', 'r') as inp:
        with open('51-2b-ss-{:03d}.CNG.swc'.format(scale), 'w') as out:
            for line in inp:
                if line.startswith('#'):
                    out.write(line)
                    if line.startswith('# Simplified soma'):
                        out.write('# Scaling factor: {}\n'.format(scale))
                else:
                    sid, tp, x, y, z, r, pid = line.split()
                    x, y, z = float(x) * xs, float(y) * xs, float(z) * xs
                    r = float(r) * rs
                    out.write(' '.join((sid, tp, str(x), str(y), str(z), str(r), pid)))
                    out.write('\n')

            
