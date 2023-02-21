#load npy
import numpy as np
# smear = "extern_Synlab_2162_82_1_MTB.czi"
smear = "extern_Synlab_2151_26_3_MTB.czi"
meta = np.load(f"TB_sample/meta_{smear}.npy")

numberoflines = int(np.max(meta[:,2]))
print(numberoflines)
# how many tiles are in every line, get maximum number of tiles in a line
maxtiles = 0
for i in range(0, numberoflines+1):
    tiles = np.where(meta[:,2] == i)[0].size
    if tiles > maxtiles:
        maxtiles = tiles

map = np.zeros((numberoflines+1, maxtiles))
# fill map with tile numbers
# get lines that have the maximum number of tiles
max_raws = []
for i in range(0, numberoflines+1):
    tiles = np.where(meta[:,2] == i)[0]

    if tiles.size == maxtiles:
        max_raws.append(i)


        map[i, :] = tiles


# go from 0 to the firs line that has the maximum number of tiles
# and fill the map with the tile numbers
for i in range(max_raws[0], 0 , -1):
    print(i, "line")
    tiles = np.where(meta[:,2] == i-1)[0]
    old_tiles = np.where(meta[:,2] == i)[0]
    num_tiles = tiles.size


    for h in range(0, maxtiles):


            if np.abs(meta[tiles[0], 1] - meta[old_tiles[0]+h, 1]) < 100:


                map[i-1, h: num_tiles+h ] = tiles

                break

# go from the last line that has the maximum number of tiles to the last line
# and fill the map with the tile numbers
for i in range(max_raws[-1], numberoflines):
    print(i, "line")
    tiles = np.where(meta[:,2] == i+1)[0]
    old_tiles = np.where(meta[:,2] == i)[0]
    num_tiles = tiles.size
    print("num_tiles", num_tiles)
    print("old_tiles size", old_tiles.size)
    print("tiles", tiles)
    if i == 8:
        # make a copy of meta
        meta_copy = np.copy(meta)
        for i in range(tiles[0], tiles[-1]):
            if i == tiles[-1]-1:
                meta[tiles[0], 1] = meta_copy[i, 1]
            meta[i+1, 1] = meta_copy[i, 1]
            
        

    for h in range(0, maxtiles):
            distance = np.abs(meta[tiles[0], 1] - meta[old_tiles[0]+h, 1])
            print(distance)
            if distance < 110:
                map[i+1, h: num_tiles+h ] = tiles
                break

#plot map
# create a random array of 1345 elements
import json 

with open(f"smears_json/{smear}.json") as f:
    data = json.load(f)

values = data["bacilli_per_single_tile"].values()

# make values an array
values = np.array(list(values))

import matplotlib.pyplot as plt
tiles_exclude = []
# plot values in the map
for l in range(0,len(values)):
    flag = False
    for i in range(0, numberoflines+1):
        for h in range(0, maxtiles):
            if map[i,h] == l and (i,h) not in tiles_exclude:
                map[i,h] = values[l]
                tiles_exclude.append((i,h))
                flag = True
                break
        if flag:
            break


# plot map
plt.imshow(map)
# add colorbar with values from 0 to 1000
plt.colorbar()
# values of colorbar go from 0 to 100
plt.clim(0, 100)
# add x and y labels
# color all pizels with value 0 black


# save map
plt.savefig(f"TB_sample/map_{smear}-.png")


# now do the same 3 dimensionally
# create a grid to plot in 3d
x = np.arange(0, numberoflines+1, 1)
y = np.arange(0, maxtiles, 1)
x, y = np.meshgrid(x, y)
# plot map in 3d
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# transpose map to plot it in 3d
map = map.T
ax.plot_surface(x, y, map)
# save map
plt.savefig(f"TB_sample/map_3d_{smear}.png")



