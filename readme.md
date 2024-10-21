## About

`gen.py` generates a region file to be put into the `region` folder for a given Minecraft world.

This utilizes the `anvil` package to write the .mca files.  Use `pip install anvil-new` to install it.

The generated region consists of planetoids, inspired by [the original](https://www.minecraftforum.net/forums/mapping-and-modding-java-edition/minecraft-tools/1260575-new-map-generator-planetoids-v1-75-now-up) map generator.

Currently, there are no known issues with this generator.  However, attempting to add other wood types or nether blocks to the generation process corrupts the region files.  I don't know how to fix this.

## Usage

Use default options:
```console
python gen.py
```

Show help:
```console
python gen.py -h
```

Command syntax:
```console
python gen.py [-h] [--region-coords REGION_COORDS] [--radius-min RADIUS_MIN] [--radius-max RADIUS_MAX] [--num-planetoids NUM_PLANETOIDS] [--out-dir OUT_DIR]
```

Example parameters (the defaults):
```console
python gen.py --region-coords "0 0" --radius-min 5 --radius-max 13 --num-planetoids 128 --out-dir ./region
```

Write the file directly into a Minecraft world (reload world to see effect, and **NOTE** that this overwrites everything in all affected chunks):
```console
python gen.py --out-dir PATH/TO/WORLD/region
```
