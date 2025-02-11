import os
import json
import numpy as np
from typing import Dict, List, Tuple, Optional


class RescanInfo:
    def __init__(self, scan_dir, split="train"):
        train_scenes, test_scenes = self.split_scenes(scan_dir)
        if split == "train":
            self.scenes = train_scenes
        else:
            self.scenes = test_scenes

    def split_scenes(self, scan_dir: str) -> Tuple[List[Dict], List[Dict]]:
        # read the "type" field from each scene and split the scenes into two lists
        # one for "train+val" and one for "test"
        scenes = json.load(open(os.path.join(scan_dir, "3RScan.json")))
        train_scenes = []
        test_scenes = []
        for scene in scenes:
            if scene["type"] == "train" or scene["type"] == "val":
                train_scenes.append(scene)
            else:
                test_scenes.append(scene)
        return train_scenes, test_scenes

    def get_scene_list(self, scene: Dict):
        # Returns a list of scan IDs and relative transformation matrices for an entire scene
        scan_id_set = [scene["reference"]]
        scan_tf_set = [np.eye(4)]
        instance_change_set = [[]]
        instance_tf_set = [[]]

        changes = []
        tfs = []
        for follow_scan in scene["scans"]:
            scan_id_set.append(follow_scan["reference"])
            if "transform" in follow_scan.keys():
                scan_tf_set.append(np.array(follow_scan["transform"]).reshape((4, 4)).T)
            else:
                scan_tf_set.append(np.eye(4))
            for change in follow_scan["rigid"]:
                if isinstance(change, int):
                    changes.append(change)
                else:
                    changes.append(change["instance_reference"])
                    tfs.append(np.array(change["transform"]).reshape((4, 4)).T)

            instance_change_set.append(changes.copy())
            instance_tf_set.append(tfs.copy())
        
        # when instance_change_set looks like this: [[], [15, 14], [15, 14, 16, 20], [15, 14, 16, 20, 12, 11]]
        # only record the incremental changes between adjacent scans
        increm_instance_change_set = [[]]
        increm_instance_tf_set = [[]]
        for i in range(1, len(instance_change_set)):
            increm_instance_change_set.append(instance_change_set[i][len(instance_change_set[i-1]):])
            increm_instance_tf_set.append(instance_tf_set[i][len(instance_tf_set[i-1]):])

        return scan_id_set, scan_tf_set, increm_instance_change_set, increm_instance_tf_set
    

    def __len__(self):
        return len(self.scenes)


    def __getitem__(self, idx):
        return self.scenes[idx]



if __name__ == "__main__":
    root = "/mnt/Backup/Dataset/3d_vsg/data/raw"
    info = RescanInfo(root, split="train")
    print(len(info))
    scene = info.scenes[1]
    for scene in info:
        scan_id_set, scan_tf_set, instance_change_set, instance_tf_set = info.get_scene_list(scene)
        print(scan_id_set)
        print(scan_tf_set)
        print(instance_change_set)
        print(instance_tf_set)
        break