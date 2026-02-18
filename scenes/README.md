# Alex robot scenes

## scenes from Molmo

Get scenes from `https://github.com/allenai/molmospaces`. Clone repo. 

Download scenes as described.

```
# python commands:
from molmo_spaces.utils.lazy_loading_utils import install_scene_with_objects_and_grasps_from_path
from molmo_spaces.molmo_spaces_constants import get_scenes 
install_scene_with_objects_and_grasps_from_path(get_scenes("ithor", "train")["train"][1])
```

Scenes were copied to dir `scenes/ithor` and `scenes/objects`.

