import sys
import argparse
import numpy as np

__version__ = '0.1'
__author__ = 'Anthony Boulmier'


def generate_trace(distribution, sigma=10, mu=50, N=10000,  verbosity=0):
    from sklearn.preprocessing import MinMaxScaler
    import matplotlib.pyplot as plt
    mms = MinMaxScaler(feature_range=(0.05, 1))

    if distribution == 'gaussian':
        s = np.random.normal(mu, sigma, N)
    elif distribution == 'gamma':
        shape, scale = 2., 2.  # mean and dispersion
        s = np.random.gamma(shape, scale, 1000)
    elif distribution == 'uniform':
        s = np.random.uniform(0, 1, N)

    #scale 0-1
    s = mms.fit_transform(s.reshape(-1, 1))

    if verbosity > 0:
        plt.hist(s, 50)
        plt.show()
    return s


parser = argparse.ArgumentParser(description='SimGRID platform generator %s' % __version__)

parser.add_argument('-f', '--filename', type=str, required=True, help='Name of the plateform file')
parser.add_argument('-n', '--nbhosts', type=int, required=True, help='Number of hosts in the platform')
parser.add_argument('-p', '--prefix', type=str, required=True, help='Prefix of the hosts')
parser.add_argument('-s', '--suffix', type=str, required=True, help='Suffix of the hosts')
parser.add_argument('-H', '--hostfile', action="store_true", help='Enable the hostfile generation')
parser.add_argument('-P', '--power', type=str, required=True, help='Power of the hosts')
parser.add_argument('-b', '--bandwidth', type=str, required=True, help='Bandwidth of the hosts\'s link')
parser.add_argument('-l', '--latency', type=str, required=True, help='Latency of the hosts')
parser.add_argument('-ld', '--loadDistribution', type=str, choices=['gaussian', 'constant', 'gamma', 'uniform'], required=True,
                    help='Generate a varying CPU load following a given probabilistic distribution')
parser.add_argument('-o', '--periodicity', type=int, required=True, help="Trace periodicity")
parser.add_argument('-v', '--verbose', action="count", help='Increase verbosity')
parser.add_argument('-c', '--corePerHost', type=str, required=True, help='Number of core per host')

args = parser.parse_args()

payload = "<?xml version='1.0'?>\n\
<!DOCTYPE platform SYSTEM \"http://simgrid.gforge.inria.fr/simgrid/simgrid.dtd\">\n\
<platform version=\"4\">\n"
if not args.loadDistribution == 'constant':
    for i in range(args.nbhosts):
        payload += '\n<trace id="%d" periodicity="%d">\n' % (i, args.periodicity)
        trace = generate_trace(distribution=args.loadDistribution, verbosity=args.verbose, N=1000)
        p = 0.0
        for load in trace:
            payload += '%f %f\n' % (p, load)
            p += float(args.periodicity)
        payload += '</trace>'

payload += '\n<AS  id="my_cluster1"  routing="Cluster">\n\
    <router id="router1"/>'
for i in range(args.nbhosts):
    hostname = "%s-%d.%s" % (args.prefix, i, args.suffix)
    payload += '\n<host id="%s" speed="%s" core="%s"/>\n\
<link id="l%d" bandwidth="%s" latency="%s"/>\n\
<host_link id="%s" up="l%d" down="l%d"/>' % (hostname, args.power, args.corePerHost, i, args.bandwidth, args.latency, hostname, i, i)

payload += '\n<backbone id="backbone1" bandwidth="%s" latency="%s"/>\n\
</AS>' % (args.bandwidth, args.latency)

if not args.loadDistribution == 'constant':
    for i in range(args.nbhosts):
        payload += '\n<trace_connect kind="SPEED" trace="%d" element="%s-%d.%s"/>' % (i, args.prefix, i, args.suffix)

payload += '\n</platform>'

with open(args.filename, 'w+') as platform:
    platform.write(payload)


if args.hostfile:
    filename = 'hostfiles/hostfile'
    with open(filename, 'w') as f:
        for i in range(args.nbhosts):
            name = "%s-%d.%s\n" % (args.prefix, i, args.suffix)
            f.write(name)


