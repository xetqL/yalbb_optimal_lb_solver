import sys
try:
    prefix = sys.argv[1]
    suffix = '.'+sys.argv[2] if sys.argv[2] != '' else ''
    filename = sys.argv[4] if len(sys.argv) > 4 else 'hostfile'
    with open(filename, 'w') as f:
        for i in range(0, int(sys.argv[3])):
            name = "%s-%d%s\n"%(prefix,i,suffix)
            f.write(name)
except Exception as e:
    print("Usage: python build_hostfile.py prefix suffix nbHosts [filename]")

