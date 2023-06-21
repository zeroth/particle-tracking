from skimage.measure import label, regionprops, regionprops_table
import numpy as np
import tifffile
import pandas as pd
import napari
import time
from tqdm import tqdm
from typing import List, Dict, Any

from particle_tracking.utils import get_statck_properties, get_tracks
from dataclasses import dataclass, field, fields

# https://www.science.org/doi/10.1126/science.abd9944
# https://github.com/tobiasjj/stepfinder/tree/master
# https://www.sciencedirect.com/science/article/pii/S2001037021002944
# https://www.sciencedirect.com/science/article/pii/016502709190118J 

def classFromDict(className, argDict):
    fieldSet = {f.name for f in fields(className) if f.init}
    filteredArgDict = {k : v for k, v in argDict.items() if k in fieldSet}
    return className(**filteredArgDict)

@dataclass(repr=True)
class Point:
    x:int
    y:int
    z:int = None
    area:float = 0
    bbox:tuple = None
    intensity_max:float = 0
    intensity_mean:float = 0
    intensity_min:float = 0
    label:int = 0
    frame:int = 0
    other:Dict[Any, Any] = None

    def __eq__(self, other):
        return (self.x, self.y, self.z) == (other.x, other.y, other.z)
    
    def __ne__(self, other):
        return not self.__eq__(other)

    def to_time_tuple(self):
        # frame/time_point, (z), y, x
        return [self.frame, self.y, self.x] if self.z is None else [self.frame, self.z, self.y, self.x]
    
    def __getitem__(self, item):
        return getattr(self, item)
    # def __repr__(self) -> str:
    #     return f'X:{self.x}, Y:{self.y}, Z:{self.z} -> {self.intensity_mean}'
    # def __lt__(self, other):
    #     return (self.x, self.y, self.z) < (other.x, other.y, other.z)
    
    # def __le__(self, other):
    #     return (self.x, self.y, self.z) <= (other.x, other.y, other.z)
    
    # def __gt__(self, other):
    #     return (self.x, self.y, self.z) > (other.x, other.y, other.z)
    
    # def __ge__(self, other):
    #     return (self.x, self.y, self.z) >= (other.x, other.y, other.z)

@dataclass
class Track:
    points:List[Point] = field(default_factory=list)
    id: int = 0
    id_key:str = field(default_factory=str)
    data_frame :pd.DataFrame = field(default_factory=pd.DataFrame)

    def __len__(self):
        return len(self.points)
    
    def __repr__(self) -> str:
        return f'id:{self.id}, len: {len(self.points)}'
    
    def init_by_dataframe(self, df:pd.DataFrame, id_key:str):
        df['label'] = df[id_key]
        self.id_key =id_key
        self.data_frame = df
        for ti, r in df.iterrows():
            self.id = int(r[id_key])
            point = classFromDict(Point, r.to_dict())
            self.points.append(point)
    def to_points_list(self):
        return list((p.to_time_tuple() for p in self.points))
    
    def to_list(self):
        # track_id, frame/time_point, (z), y, x
        return list(([self.id, ] + p.to_time_tuple() for p in self.points))
    
    def to_list_by_key(self, key):
        return list((p[key] for p in self.points))

@dataclass
class PointsPool:
    points:List[Point] = field(default_factory=list)

    def __len__(self):
        return len(self.points)

    def find_by_frame(self, frame):
        for p in self.points:
            if p.frame == frame:
                return p
        return None

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
    accepted_track_count = 0
    tracks = []
    points = []
    track_objs = []
    for track_id in track_ids:
        if len(list(tracked[tracked['particle'] == track_id]['frame'])) >= 200:
            accepted_track_count +=1
            track = Track()
            track.init_by_dataframe(tracked[tracked['particle'] == track_id].copy().reset_index(drop=True), 'particle')
            # for ti, r in track.iterrows():
            track_objs.append(track)
            tracks += track.to_list()
            points += track.to_points_list()
    print(f'totlat time require:  {time.time() - st} sec')

    print(f"Total tracks :  {accepted_track_count}")
    print(f"Total number of tracks : {len(tracks)}")
    print(f"tracks[0] : {tracks[0]}, type : {type(tracks[0])}")


    print(f"Total number of points : {len(points)}")
    print(f"point[0] : {points[0]}, type : {type(points[0])}")

    print(f"Intensity tracks[0] : {track_objs[0].to_list_by_key('intensity_mean')}")
    # print(len(tracks))
    # track = tracks[0]
    # for p in track.points:
    #     print(p)
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

    
    # print(f"points len = {len(points)}")
    # points = np.array(points)
    # tracks = np.array(tracks)
    viewer = napari.view_image(images, name='raw_particles', rgb=False)
    viewer.add_labels(masks, name='particles')
    #viewer.add_labels(mask_label, name='particles_labels')
    viewer.add_points(points, name="points", size=2)
    viewer.add_tracks(tracks, name="Trackes", tail_width =2, tail_length=5)
    napari.run()


if __name__ == "__main__":
    main()