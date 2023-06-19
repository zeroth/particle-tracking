from skimage.measure import label, regionprops, regionprops_table
import numpy as np
import tifffile
import pandas as pd
import napari
import time
from tqdm import tqdm

from particle_tracking.utils import get_statck_properties, get_tracks
from dataclasses import dataclass, field



def main():
    masks_path = "D:/Data/Sudipta/Arpan/position_test/mask.tif"
    images_path = "D:/Data/Sudipta/Arpan/send-1.tif"

    images = tifffile.imread(images_path)
    masks = tifffile.imread(masks_path)

    assert images.shape == masks.shape
    st = time.time()
    main_pd_frame = get_statck_properties(masks=masks, images=images, show_progress=True)
    
    with pd.option_context('display.max_rows', 200, 'display.max_columns', None):  # more options can be specified also
        print(main_pd_frame)
    

    # trackpy.quiet(True)
    # tracked = trackpy.link(f=main_pd_frame,search_range=2, memory=0)
    tracked = get_tracks(main_pd_frame)
    track_ids = tracked['particle'].unique()
    tracks = []
    for track_id in track_ids:
        if len(list(tracked[tracked['particle'] == track_id]['frame'])) >= 1:
            track = tracked[tracked['particle'] == track_id].copy().reset_index(drop=True)
            for ti, r in track.iterrows():
                tracks.append([r['particle'], r['frame'], r['y'], r['x']])

    # with pd.option_context('display.max_rows', 200, 'display.max_columns', None):  # more options can be specified also
    #     print(tracked)
    # s = tracked.groupby("particle").size()
    # print(f'Size : {s}\ntype: {type(s)}')
    # print(s)
    
    # time_str = time.strftime("%d_%m_%Y_%H_%M_%S", time.gmtime())
    # s.to_csv(f"D:/Data/Sudipta/Arpan/op/size_{time_str}.csv")

    # points = np.array(shape=[images.shape[0]])
    # for index, row in tracked.iterrows():
    #     points.append([row['y'],  row['x']])

    print(f'totlat time require:  {time.time() - st} sec')
    # print(f"points len = {len(points)}")
    # points = np.array(points)
    viewer = napari.view_image(images, name='raw_particles', rgb=False)
    viewer.add_labels(masks, name='particles')
    #viewer.add_labels(mask_label, name='particles_labels')
    # viewer.add_points(points, name="points", size=2)
    viewer.add_tracks(tracks, name="Trackes", tail_width =2, tail_length=5)
    napari.run()


if __name__ == "__main__":
    main()