# # import dask.array as da
# import os
# # import tifffile
# # from torch.utils.data import DataLoader
# # path = "D:/Data/Sudipta/Arpan/ML/Data/exp1"
# # arr = da.from_zarr(os.path.join(path, "ord_data.zarr"))
# # os.makedirs(os.path.join(path, "org"), exist_ok=True)
# # for i in range(arr.shape[0]):
# #     tifffile.imwrite(os.path.join(path, "org",f"org_{i}.tiff"), arr[i])

# # import numpy as np

# import sys
# import hashlib

# def main():
#     # root = "."
#     # SimpleOxfordPetDataset.download(root)
#     # train_dataset = SimpleOxfordPetDataset(root, "train")
#     # print(train_dataset[0]['image'].shape)
#     # n_cpu = os.cpu_count()
#     # train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=1)
#     # print(f"Train size: {len(train_dataset)}")
#     # for batch in train_dataloader:
#     #     print(batch['image'].shape)
#     # BUF_SIZE is totally arbitrary, change for your app!
#     BUF_SIZE = 65536  # lets read stuff in 64kb chunks!

#     md5 = hashlib.md5()
#     sha1 = hashlib.sha1()

#     with open(sys.argv[1], 'rb') as f:
#         while True:
#             data = f.read(BUF_SIZE)
#             if not data:
#                 break
#             md5.update(data)
#             sha1.update(data)

#     print("MD5: {0}".format(md5.hexdigest()))
#     print("SHA1: {0}".format(sha1.hexdigest()))
    

# if __name__ == '__main__':
#     main()
#--------------

import numpy as np
from skimage import data
import napari


blobs = np.stack(
    [
        data.binary_blobs(
            length=128, blob_size_fraction=0.05, n_dim=3, volume_fraction=f
        )
        for f in np.linspace(0.05, 0.5, 10)
    ],
    axis=0,
)
viewer = napari.view_image(blobs.astype(float))
point_properties = {
    'good_point': np.array([True, True, False, True, False]),
    'confidence': np.array([0.99, 0.8, 1, 2, 3])
}
# add the points
points = np.array(
    [
        [0, 0, 100, 100],
        [0, 0, 50, 120],
        [1, 0, 100, 40],
        [2, 10, 110, 100],
        [9, 8, 80, 100],
    ], dtype=float
)
viewer.add_points(
    points, properties=point_properties, size=[0, 6, 10, 10], face_color='blue', out_of_slice_display=True
)

if __name__ == '__main__':
    napari.run()