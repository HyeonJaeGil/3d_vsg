import os
import json
import numpy as np
from typing import Dict, List, Tuple, Optional
import open3d as o3d
from collections import defaultdict
import networkx as nx


class AnnotatedSceneGraphLoader:
    def __init__(self, scan_dir):
        self.scan_dir = scan_dir
        self.check_file_existence(scan_dir)
        self.objects_dict, self.relationships_dict = self.get_dataset_files(scan_dir)
        self.scan_ids = self.extract_scan_ids()
        self.check_semseg_existence(required_files=["semseg.v2.json"])
        self.scene_graphs = {scan_id: {} for scan_id in self.scan_ids}

    def check_file_existence(self, scan_dir):
        # Check if all required files exist in the specified directory
        required_folders = ["3DSSG", "semantic_segmentation_data"]
        required_files = ["3DSSG/objects.json", "3DSSG/relationships.json"]
        for folder in required_folders:
            if not os.path.isdir(os.path.join(scan_dir, folder)):
                raise FileNotFoundError(f"Directory {folder} not found in {scan_dir}")
        for file in required_files:
            if not os.path.exists(os.path.join(scan_dir, file)):
                raise FileNotFoundError(f"File {file} not found in {scan_dir}")
        return True
    
    def extract_scan_ids(self):
        # read self.objects_dict and self.relationships_dict to get the scan ids
        # assert that the scan ids are the same in both dictionaries
        scan_ids_objects = set(self.objects_dict.keys())
        scan_ids_relationships = set(self.relationships_dict.keys())

        # get the union of the scan ids in both dictionaries
        scan_ids_common = scan_ids_objects.intersection(scan_ids_relationships)

        # get the difference of the scan ids in both dictionaries
        scan_ids_objects_only = scan_ids_objects.difference(scan_ids_common)
        scan_ids_relationships_only = scan_ids_relationships.difference(scan_ids_common)

        # print(f"Scan IDs in objects dictionary: {scan_ids_objects_only}")
        # print(f"Scan IDs in relationships dictionary: {scan_ids_relationships_only}")

        return sorted(scan_ids_common)

    def check_semseg_existence(self, required_files=[]):
        # Check if semantic segmentation files exist for all scans
        for scan_id in self.scan_ids:
            scan_dir = os.path.join(self.scan_dir, "semantic_segmentation_data", scan_id)
            for file in required_files:
                if not os.path.exists(os.path.join(scan_dir, file)):
                    raise FileNotFoundError(f"File {file} not found in {scan_dir}")
        return True

    def format_scan_dict(self, unformated_dict: Dict, attribute: str) -> Dict:
        # Format raw dictionary of object nodes for all scenes
        scan_list = unformated_dict["scans"]
        formatted_dict = {}
        for scan in scan_list:
            formatted_dict[scan["scan"]] = scan[attribute]
        return formatted_dict

    def get_dataset_files(self, scan_dir):
        object_data = json.load(open(os.path.join(scan_dir, "3DSSG", "objects.json")))
        relationship_data = json.load(open(os.path.join(scan_dir, "3DSSG", "relationships.json")))
        objects_dict = self.format_scan_dict(object_data, "objects")
        relationships_dict = self.format_scan_dict(relationship_data, "relationships")
        return objects_dict, relationships_dict

    def get_ply_vertices(self, scan_dir, scan_id):
        filename = os.path.join(scan_dir, "semantic_segmentation_data", scan_id, "labels.instances.annotated.v2.ply")
        with open(filename, "r") as f:
            lines = f.readlines()

        # Parse header to find vertex start
        header_ended = False
        vertex_lines = []
        for i, line in enumerate(lines):
            if line.strip() == "end_header":
                header_ended = True
                vertex_start_idx = i + 1
                break

        if not header_ended:
            raise ValueError("Invalid PLY file: No 'end_header' found.")

        # Extract vertex data
        vertices = []
        for line in lines[vertex_start_idx:]:
            tokens = line.split()
            if len(tokens) != 11:  # Ensure correct format
                continue
            x, y, z = map(float, tokens[:3])
            r, g, b = map(int, tokens[3:6])
            objectId = int(tokens[6])
            globalId = int(tokens[7])
            NYU40 = int(tokens[8])
            Eigen13 = int(tokens[9])
            RIO27 = int(tokens[10])

            vertices.append([x, y, z, r, g, b, objectId, globalId, NYU40, Eigen13, RIO27])

        # Convert to NumPy array
        vertices = np.array(vertices, dtype=np.float32)

        # Group by objectId
        grouped_data = defaultdict(lambda: {"xyz": [], "color": [], "globalId": [], "NYU40": [], "Eigen13": [], "RIO27": []})
        
        for v in vertices:
            obj_id = int(v[6])
            grouped_data[obj_id]["xyz"].append(v[:3])
            grouped_data[obj_id]["color"].append(v[3:6])
            grouped_data[obj_id]["globalId"].append(v[7])
            grouped_data[obj_id]["NYU40"].append(v[8])
            grouped_data[obj_id]["Eigen13"].append(v[9])
            grouped_data[obj_id]["RIO27"].append(v[10])

        # Convert lists to NumPy arrays for efficient processing
        for obj_id in grouped_data:
            grouped_data[obj_id]["xyz"] = np.array(grouped_data[obj_id]["xyz"], dtype=np.float32)
            grouped_data[obj_id]["color"] = np.array(grouped_data[obj_id]["color"], dtype=np.uint8)
            grouped_data[obj_id]["globalId"] = np.array(grouped_data[obj_id]["globalId"], dtype=np.uint16)
            grouped_data[obj_id]["NYU40"] = np.array(grouped_data[obj_id]["NYU40"], dtype=np.uint8)
            grouped_data[obj_id]["Eigen13"] = np.array(grouped_data[obj_id]["Eigen13"], dtype=np.uint8)
            grouped_data[obj_id]["RIO27"] = np.array(grouped_data[obj_id]["RIO27"], dtype=np.uint8)
        
        # merge the array of color, globalId, NYU40, Eigen13, RIO27 to a single array if the values are the same
        for obj_id in grouped_data:
            color = grouped_data[obj_id]["color"]
            globalId = grouped_data[obj_id]["globalId"]
            NYU40 = grouped_data[obj_id]["NYU40"]
            Eigen13 = grouped_data[obj_id]["Eigen13"]
            RIO27 = grouped_data[obj_id]["RIO27"]
            if np.all(color == color[0]):
                color = grouped_data[obj_id]["color"][0]
                grouped_data[obj_id]["color_hex"] = "#{:02x}{:02x}{:02x}".format(color[0], color[1], color[2])
            if np.all(globalId == globalId[0]):
                grouped_data[obj_id]["globalId"] = globalId[0]
            if np.all(NYU40 == NYU40[0]):
                grouped_data[obj_id]["NYU40"] = NYU40[0]
            if np.all(Eigen13 == Eigen13[0]):
                grouped_data[obj_id]["Eigen13"] = Eigen13[0]
            if np.all(RIO27 == RIO27[0]):
                grouped_data[obj_id]["RIO27"] = RIO27[0]

        return grouped_data

    def get_semseg(self, scan_dir, scan_id):
        sem_seg_data = json.load(open(os.path.join(scan_dir, "semantic_segmentation_data", scan_id, "semseg.v2.json")))
        sem_seg_dict = sem_seg_data["segGroups"]
        return sem_seg_dict

    def get_semseg_with_pointcloud(self, scan_dir, scan_id):
        sem_seg_dict = self.get_semseg(scan_dir, scan_id)
        ply_data = self.get_ply_vertices(scan_dir, scan_id)

        # add xyz field in ply_data to sem_seg_dict
        for key in ply_data:
            object_id = key
            matched_objects = [seg for seg in sem_seg_dict if seg["id"] == object_id]
            if len(matched_objects) == 0:
                print(f"Object {object_id} not found in semseg dict")
                continue
            object = matched_objects[0]
            object["xyz"] = ply_data[key]["xyz"]
        return sem_seg_dict

    def get_annotated_ply(self, scan_dir, scan_id):
        filename = os.path.join(scan_dir, "semantic_segmentation_data", scan_id, "labels.instances.annotated.v2.ply")
        o3d_ply = o3d.io.read_point_cloud(filename)
        return o3d_ply

    def build_scene_graph(self, scan_id: str, load_points=False, graph_out=False) -> Tuple:
        # Returns a scene graph from raw data, including:
        #   - Nodes: objects with relevant attributes
        #   = Edges: relationships between objects

        nodes = self.objects_dict[scan_id]
        edges = self.relationships_dict[scan_id]

        semseg = self.get_semseg_with_pointcloud(self.scan_dir, scan_id) if load_points else self.get_semseg(self.scan_dir, scan_id)
        sem_seg_dict = {int(object["id"]): object for object in semseg}


        # Reformat node dictionary, include only relevant attributes, and add location
        nodes_dict = {}
        # input_node_list = []
        for node in nodes:
            node_copy = node.copy()
            id = int(node["id"])
            att_dict = {"label": node_copy.pop("label", None), "affordances": node_copy.pop("affordances", None),
                        "attributes": node_copy.pop("attributes", None), "global_id": node_copy.pop("global_id", None),
                        "color": node_copy.pop("ply_color", None)}

            # if object_pos_list is not None:
            #     att_dict["attributes"]["location"] = torch.tensor(np.clip(object_pos_list[id], -100, 100)).to(torch.float32)

            att_dict["centroid"] = sem_seg_dict[id]["obb"]["centroid"]
            if load_points:
                att_dict["pointcloud"] = sem_seg_dict[id]["xyz"]

            att_dict["attributes"].pop("lexical", None)
            # input_node_list.append((id, att_dict))
            nodes_dict[id] = att_dict

        # Can output a networkx Graph object for visualization purposes
        if False:
            graph = nx.Graph()
            graph.add_nodes_from(input_node_list)
            for edge in edges:
                graph.add_edge(edge[0], edge[1])
        else:
            graph = None

        return graph, nodes_dict, edges

    # def __getitem__(self, scan_id):
    #     # Get scene graph data for a single scan
    #     sem_seg_dict = self.get_semseg(self.scan_dir, scan_id)
    #     ply_data, o3d_ply = self.get_ply_vertices(self.scan_dir, scan_id)
    #     return sem_seg_dict, ply_data, o3d_ply

def hex_to_rgb(hex):
    hex = hex.lstrip("#")
    return tuple(int(hex[i:i+2], 16)/255. for i in (0, 2, 4))

def rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))

def numpy_to_o3d_pointcloud(points, color="#000000"):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color(hex_to_rgb(color))
    return pcd

if __name__ == "__main__":
    root = "/mnt/Backup/Dataset/3d_vsg/data"
    scan_dir = os.path.join(root, "raw")    
    sg_loader = AnnotatedSceneGraphLoader(scan_dir)
    for scan_id in ["ddc737b3-765b-241a-9c35-6b7662c04fc9"]:
        graph, nodes, edges = sg_loader.build_scene_graph(scan_id, load_points=True)
        annotated_ply = sg_loader.get_annotated_ply(scan_dir, scan_id)
        print("Length of nodes:", len(nodes))
        for k, v in nodes.items():
            print(k, v.keys())
        # for item in edges:
        #     o1, o2, r, s = item
        #     print(nodes[o1]["label"], s, nodes[o2]["label"])

        for item in edges:
            o1, o2, r, s = item
            print(nodes[o1]["label"], s, nodes[o2]["label"])
            pcd1 = numpy_to_o3d_pointcloud(nodes[o1]["pointcloud"], color=nodes[o1]["color"])
            pcd2 = numpy_to_o3d_pointcloud(nodes[o2]["pointcloud"], color=nodes[o2]["color"])
            o3d.visualization.draw_geometries([annotated_ply.paint_uniform_color([0.5, 0.5, 0.5]), pcd1, pcd2])
            # break

        break