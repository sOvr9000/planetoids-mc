
import os
import numpy as np
import argparse
from typing import Tuple, Iterable, List, Dict, Any
from math import ceil
from anvil import Block, EmptyRegion, RawSection
from functools import lru_cache



@lru_cache(maxsize=50)
def generate_layered_sphere(radii: Tuple[int, ...]) -> np.ndarray:
    # Generate only in one of eight quadrants
    radii = list(radii)
    radii.sort() # for iterating over layers in the correct order
    size = 2 * ceil(radii[-1]) + 1
    center = size // 2
    sphere = np.zeros((size, size, size), dtype=np.uint8)
    for x in range(center + 1):
        for y in range(center + 1):
            for z in range(center + 1):
                dist = x * x + y * y + z * z
                for i, r in enumerate(radii):
                    if dist <= r * r:
                        layer = i + 1
                        sphere[center + x, center + y, center + z] = layer
                        sphere[center - x, center + y, center + z] = layer
                        sphere[center + x, center - y, center + z] = layer
                        sphere[center - x, center - y, center + z] = layer
                        sphere[center + x, center + y, center - z] = layer
                        sphere[center - x, center + y, center - z] = layer
                        sphere[center + x, center - y, center - z] = layer
                        sphere[center - x, center - y, center - z] = layer
                        break
    return sphere

# print(generate_layered_sphere((1, 2)))
# input()

def build_sphere(
    region_arr: np.ndarray,
    radii: Tuple[int, ...],
    center: Tuple[int, int, int],
) -> Iterable[Tuple[int, int, int, np.uint8]]:
    '''
    Iterate over the block positions that have changed and the layer indices to which they changed.
    '''
    center_x, center_y, center_z = center
    sphere = generate_layered_sphere(radii)
    lower_x = center_x - sphere.shape[0] // 2
    upper_x = lower_x + sphere.shape[0]
    lower_y = center_y - sphere.shape[1] // 2
    upper_y = lower_y + sphere.shape[1]
    lower_z = center_z - sphere.shape[2] // 2
    upper_z = lower_z + sphere.shape[2]
    r = region_arr[
        lower_x:upper_x,
        lower_y:upper_y,
        lower_z:upper_z,
    ]
    changed = np.logical_and(r == 0, sphere > 0)
    region_arr[
        lower_x:upper_x,
        lower_y:upper_y,
        lower_z:upper_z,
    ] = np.where(
        r == 0,
        sphere,
        r,
    )
    for x in range(sphere.shape[0]):
        for y in range(sphere.shape[1]):
            for z in range(sphere.shape[2]):
                if changed[x, y, z] and sphere[x, y, z] > 0:
                    yield (
                        lower_x + x,
                        lower_y + y,
                        lower_z + z,
                        sphere[x, y, z],
                    )

def choice(obj: Any) -> Any:
    if isinstance(obj, Dict):
        # select by weight
        weights = np.array(list(obj.values())).astype(float)
        weights /= weights.sum()
        return choice(np.random.choice(list(obj.keys()), p=weights))
    if isinstance(obj, List):
        # select randomly
        return choice(obj[np.random.randint(len(obj))])
    # return the object otherwise
    return obj

def generate_planetoids(
    region_coords: Tuple[int, int], # x, z

    planetoid_radius_min: int = 5, # the minimum possible radius of each planetoid, inclusive
    planetoid_radius_max: int = 13, # the maximum possible radius of each planetoid, inclusive
    num_planetoids: int = 64, # the number of planetoids to generate per region
    
    palette: List[Block] = None, # the block palette to use for the planetoids
    planetoid_type_weights: Dict[str, float] = None,

    out_region_path: str = './region', # the path to the generated region files
):
    if palette is None:
        palette = [
            Block('minecraft', 'air'),
            Block('minecraft', 'stone'),
            Block('minecraft', 'cobblestone'),
            Block('minecraft', 'iron_ore'),
            Block('minecraft', 'coal_ore'),
            Block('minecraft', 'gold_ore'),
            Block('minecraft', 'lapis_ore'),
            Block('minecraft', 'redstone_ore'),
            Block('minecraft', 'diamond_ore'),
            Block('minecraft', 'emerald_ore'),
            Block('minecraft', 'copper_ore'),
            # Block('minecraft', 'glowstone'),
            Block('minecraft', 'glass'),
            Block('minecraft', 'lava'),
            Block('minecraft', 'water'),
            Block('minecraft', 'oak_log'),
            Block('minecraft', 'oak_leaves'),
            # Block('minecraft', 'dark_oak_log'),
            # Block('minecraft', 'dark_oak_leaves'),
            # Block('minecraft', 'spruce_log'),
            # Block('minecraft', 'spruce_leaves'),
            # Block('minecraft', 'birch_log'),
            # Block('minecraft', 'birch_leaves'),
            # Block('minecraft', 'acacia_log'),
            # Block('minecraft', 'acacia_leaves'),
            # Block('minecraft', 'jungle_log'),
            # Block('minecraft', 'jungle_leaves'),
            # Block('minecraft', 'mangrove_log'),
            # Block('minecraft', 'mangrove_leaves'),
            # Block('minecraft', 'cherry_log'),
            # Block('minecraft', 'cherry_leaves'),
            # Block('minecraft', 'netherrack'),
            # Block('minecraft', 'nether_bricks'),
            # Block('minecraft', 'nether_quartz_ore'),
            # Block('minecraft', 'crimson_hyphae'),
            # Block('minecraft', 'warped_hyphae'),
        ]
    if planetoid_type_weights is None:
        planetoid_type_weights = {
            'tree': 12,
            # 'nether_tree': 2,
            'iron': 4,
            'coal': 5,
            'gold': 1,
            'lapis_lazuli': 2,
            'redstone': 2,
            'diamond': 1,
            'emerald': 0.75,
            'copper': 5,
            'lava': 2,
            'water': 3,
            # 'glowstone': 2,
            # 'nether_quartz': 1,
        }
    palette_name_index_map = {
        block.name(): i
        for i, block in enumerate(palette)
    }
    # print(palette_name_index_map)
    
    planetoid_materials = {
        'tree': [
            ('oak_log', 'oak_leaves'),
            # ('dark_oak_log', 'dark_oak_leaves'),
            # ('spruce_log', 'spruce_leaves'),
            # ('birch_log', 'birch_leaves'),
            # ('acacia_log', 'acacia_leaves'),
            # ('jungle_log', 'jungle_leaves'),
            # ('mangrove_log', 'mangrove_leaves'),
            # ('cherry_log', 'cherry_leaves'),
        ],
        # 'nether_tree': [
        #     (['netherrack', 'nether_bricks'], 'crimson_hyphae'),
        #     (['netherrack', 'nether_bricks'], 'warped_hyphae'),
        # ],
        # 'glowstone': [
        #     # (['nether_bricks', 'netherrack'], 'glowstone'),
        #     ('glowstone', 'netherrack'),
        # ],
        # 'nether_quartz': [
        #     (['netherrack', 'nether_bricks'], 'nether_quartz_ore'),
        # ],
        'lava': [
            ('lava', 'glass'),
        ],
        'water': [
            ('water', 'glass'),
        ],
        'iron': [
            ('iron_ore', ['stone', 'cobblestone']),
        ],
        'copper': [
            ('copper_ore', ['stone', 'cobblestone']),
        ],
        'coal': [
            ('coal_ore', ['stone', 'cobblestone']),
        ],
        'gold': [
            ('gold_ore', ['stone', 'cobblestone']),
        ],
        'lapis_lazuli': [
            ('lapis_ore', ['stone', 'cobblestone']),
        ],
        'redstone': [
            ('redstone_ore', ['stone', 'cobblestone']),
        ],
        'diamond': [
            ('diamond_ore', ['stone', 'cobblestone']),
        ],
        'emerald': [
            ('emerald_ore', ['stone', 'cobblestone']),
        ],
    }
    
    os.makedirs(out_region_path, exist_ok=True)
    rx, rz = region_coords
    region_arr = np.zeros((2, 512, 256, 512), dtype=np.uint8)
    # print('region array size:', region_arr.nbytes // 1024 // 1024, 'MB')
    types_generated = {t: 0 for t in planetoid_type_weights.keys()}
    for _ in range(num_planetoids):
        planetoid_radius = np.random.randint(planetoid_radius_min, planetoid_radius_max + 1)
        x, z = np.random.randint(planetoid_radius, 511 - planetoid_radius, 2)
        y = np.random.randint(planetoid_radius, 255 - planetoid_radius)
        planetoid_type = choice(planetoid_type_weights)
        materials = choice(planetoid_materials[planetoid_type])
        types_generated[planetoid_type] += 1
        for x, y, z, layer in build_sphere(
            region_arr[0],
            (planetoid_radius, planetoid_radius - np.random.randint(2, 3)),
            (x, y, z),
        ):
            material = choice(materials[layer - 1])
            if ':' not in material:
                material = 'minecraft:' + material
            region_arr[1, x, y, z] = palette_name_index_map[material]
    print(f'Region {rx, rz} generated')
    print(f'Planetoid types generated: {types_generated}')
    region = EmptyRegion(rx, rz)
    for _cx in range(32):
        cx = rx * 32 + _cx
        for _cz in range(32):
            cz = rz * 32 + _cz
            for y in range(16):
                blocks = region_arr[1,
                    cx * 16 : cx * 16 + 16,
                    y * 16 : y * 16 + 16,
                    cz * 16 : cz * 16 + 16,
                ].transpose((1, 2, 0)).flatten().tolist()
                region.add_section(
                    RawSection(
                        y,
                        blocks,
                        palette,
                    ),
                    cx,
                    cz,
                )
    print(f'Region {rx, rz} indexed')
    fpath = f'{out_region_path}/r.{rx}.{rz}.mca'
    region.save(fpath)
    print(f'Region {rx, rz} saved to {fpath}')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--region-coords',
        type=str,
        default='0 0',
        help='(string) XZ coordinates of the region to generate',
    )
    
    parser.add_argument(
        '--radius-min',
        type=int,
        default=5,
        help='(int) The minimum possible radius of each planetoid, inclusive',
    )
    
    parser.add_argument(
        '--radius-max',
        type=int,
        default=13,
        help='(int) The maximum possible radius of each planetoid, inclusive',
    )
    
    parser.add_argument(
        '--num-planetoids',
        type=int,
        default=128,
        help='(int) The number of planetoids to generate',
    )
    
    parser.add_argument(
        '--out-dir',
        type=str,
        default='./region',
        help='(string) The directory under which the generated region file is saved',
    )
    
    parsed = parser.parse_args()
    region_coords = tuple(map(int, parsed.region_coords.split()))

    generate_planetoids(
        region_coords=region_coords,
        planetoid_radius_min=parsed.radius_min,
        planetoid_radius_max=parsed.radius_max,
        num_planetoids=parsed.num_planetoids,
        out_region_path=parsed.out_dir,
    )

