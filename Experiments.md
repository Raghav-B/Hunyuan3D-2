Interface Changes:
- Add multiview texgen resolution option to interface
- Add checkboxes for different parts of the multiview


Algorithm/Pipeline Changes:
- Increased delight model resolution
- Increase multiview texgen resolution to 768 from 512



Experiments to try:
- Turn off delight model. I'm doing this to prevent the orange burn effect from the delight model.
- Does the delight model indeed cause the orange effect in the actual hunyuan 3d repo?
- Does improving the bake_from_multiview() method make any differences?
    Perhaps the method or the bake_exp


Issues:
- Face detail is lost in the multiview generator. Not sure why.
- The generated mesh is "pixelated/voxelate". Probably won't matter with the textures enabled.
- Why does using a batch size of 1 mess things up?


Is there a difference with how multiview is handled in ComfyUI? It seems a bit different. My multiview looks VERY ugly for some reason.
It seems to get worse above certain resolutions




