import pstats
from pstats import SortKey

sts = pstats.Stats('profile.log')
print("="*30)
sts.sort_stats(SortKey.TIME).print_stats(20)
print("="*30)
sts.sort_stats(SortKey.CUMULATIVE).print_stats(40)
