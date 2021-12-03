import os
from box import Box
import rioxarray as rxr
import earthpy as et

import rasterio
from rasterio.plot import show as sh
import yaml


class User:
    id = 1
    name: str = 'John Doe'

    def pr():
        print(id, name)


user = User()
user.pr()
print(user.id)
