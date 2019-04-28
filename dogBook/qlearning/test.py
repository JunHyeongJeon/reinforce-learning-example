from collections import defaultdict

q_table = defaultdict(lambda: [1.0, 2.0, 3.0, 4.0])
print(dict(q_table))

print(q_table[99][0])