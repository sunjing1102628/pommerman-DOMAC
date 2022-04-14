import pstats
p=pstats.Stats('./restates')
p.strip_dirs().sort_stats('cumtime').print_stats()