import os
import sys
from tqdm import tqdm


class tif2png():

    '''
    object to cut large raw image to preferred patch_size

    '''

    def __init__(self):
        self.counter = 0

    def __call__(self,
                 config,
                 paths,
                 naip,
                 sentinel_rgb,
                 patch_size,
                 res_naip,
                 res_sentinel,
                 res_out=224):

        nr_images_x = int(naip.size[0]/(patch_size/res_naip))
        nr_images_y = int(naip.size[1]/(patch_size/res_naip))
        patch_size_naip = patch_size / res_naip
        patch_size_sentinel = patch_size / res_sentinel

        if not os.path.exists(paths['sat']):
            os.makedirs(paths['sat_rgb'])

        if not os.path.exists(paths['drone']):
            os.makedirs(paths['drone'])

        for i in tqdm(range(0, nr_images_x)):
            for j in range(0, nr_images_y):

                naip_crop = naip.crop((int(j*patch_size_naip), int(i*patch_size_naip),
                                       int(patch_size_naip + j*patch_size_naip), int(patch_size_naip+i*patch_size_naip))).resize((res_out, res_out))

                sentinel_rgb_crop = sentinel_rgb.crop((int(j*patch_size_sentinel), int(i*patch_size_sentinel),
                                                       int(patch_size_sentinel+j * patch_size_sentinel), int(patch_size_sentinel+i*patch_size_sentinel))).resize((res_out, res_out))

                if config.model_name == 'MetricLearning':
                    os.makedirs(paths['drone'] + '/' + str(self.counter))
                    sentinel_rgb_crop.save(
                        paths['sat'] + '/' + str(self.counter)+'/sentinel_'+str(self.counter)+'.png')

                    naip_crop.save(
                        paths['drone'] + '/' + str(self.counter)+'/drone_'+str(self.counter)+'.png')
                else:
                    sentinel_rgb_crop.save(
                        paths['sat'] + '/satellite_' + str(self.counter)+'.png')

                    naip_crop.save(paths['drone'] + '/drone_' +
                                   str(self.counter)+'.png')

                self.counter += 1
