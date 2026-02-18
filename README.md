# Alex robot

Developing software and control for a humanoid robot.

![](images/alex.png)




## Scenes

Get scenes from https://github.com/allenai/molmospaces. Clone repo. 

Download scenes as described"

```
# python commands:
from molmo_spaces.utils.lazy_loading_utils import install_scene_with_objects_and_grasps_from_path
from molmo_spaces.molmo_spaces_constants import get_scenes 
install_scene_with_objects_and_grasps_from_path(get_scenes("ithor", "train")["train"][1])
```

Scenes were copied to dir `scenes/ithor` and `scenes/objects`.


## Run

###  MuJoCo

You can run alex model by dragging / dropping [this file](scenes/alex-scenes/scene_alex_v1_full_body_mjx_room1.xml) into MuJoCo. Or any file in that directory `scenes/alex-scenes/`



