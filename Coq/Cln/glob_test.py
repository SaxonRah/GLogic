import glob, os
files = sorted(glob.glob('*.glob'))
print(f'Found {len(files)} glob files:')
for f in files:
    size = os.path.getsize(f)
    print(f'  {f} ({size} bytes)')
print()
if files:
    f = files[0]
    print(f'=== First 50 lines of {f} ===')
    with open(f, 'r', encoding='utf-8', errors='replace') as fh:
        for i, line in enumerate(fh):
            if i >= 50: break
            print(f'{i:3d}: {repr(line.rstrip())}')