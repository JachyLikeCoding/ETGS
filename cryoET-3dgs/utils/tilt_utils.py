import numpy as np
from scene.tilts import Tilt
from utils.general_utils import PILtoTorch


WARNED = False

def loadTilt(args, id, tilt_info, resolution_scale):
    orig_w, orig_h = tilt_info.image.size

    if args.resolution in [1,2,4,8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:
        if args.resolution == -1:
            if orig_w > 4096:
                global WARNED
                if not WARNED:
                    print("[INFO] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K. \n"
                          "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 4096
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution
        
        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h/scale))

    resized_image = PILtoTorch(tilt_info.image, resolution)

    gt_image = resized_image[:3, ...]

    return Tilt(tilt_id=tilt_info.uid, R=tilt_info.R, T=tilt_info.T, image=gt_image, 
                image_name=tilt_info.image_name, uid=id, data_device=args.data_device)




def tiltList_from_tiltInfos(tilt_infos, resolution_scale, args):
    tilt_list = []

    for id, t in enumerate(tilt_infos):
        tilt_list.append(loadTilt(args, id, t, resolution_scale))
    
    return tilt_list


def tilt_to_JSON(id, tilt:Tilt):
    Rt = np.zeros((4,4))
    Rt[:3, :3] = tilt.R.transpose()
    Rt[:3, 3] = tilt.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    tilt_entry = {
        'id': id,
        'img_name': tilt.image_name,
        'width': tilt.width,
        'height': tilt.height,
        'position': pos.tolist(),
        'rotation':serializable_array_2d,
    }
    return tilt_entry