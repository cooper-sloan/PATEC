#Map each user to list of tuples (MovieID, rating)
#Divide number of entries in map(# of users) by 500
#Randomly write ^ number of user to new data file - 500 partitions of data, with the same amount of users per each partition

import math
import numpy as np
def dump_user_map(user_map, index):
  t = []
  while len(user_map.keys()) > 0:
    curr_user, rating_list = user_map.popitem()
    for rating in rating_list:
      t.append("%s\t%s\t%s\n" % (curr_user, rating[0], rating[1]))

  with open('partition/train_%s.txt' % index, 'a') as f:
    f.write(''.join(t))

user_map = {}
user_count = 0
perm = np.random.permutation(500)

with open('nf.train.txt', 'r') as f:

  for line in f:
    parts = line.strip().split('\t')
    if len(parts) < 3:
      raise ValueError('Encountered badly formatted line')

    #If user not already in the map add them
    if not parts[0] in user_map:
      user_count += 1
      if user_count % 10000 == 0:
          print(user_count)
      user_map[parts[0]] = []
    user_map[parts[0]].append((parts[1], parts[2]))
    if len(user_map.keys()) >= 100:
      last, perm = perm[-1], perm[:-1]
      dump_user_map(user_map, last)
    if len(perm) == 0:
      perm = np.random.permutation(500)
  if len(user_map.keys()) > 0:
      last, perm = perm[-1], perm[:-1]
      dump_user_map(user_map, last)

