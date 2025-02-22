import json
import math
import random
from dataclasses import dataclass, field
from typing import Any, Optional, Union

import h5py
import networkx as nx
import numpy as np
from ai2thor.controller import Controller


def calculate_boundingbox_size(boundingbox: list) -> float:
    x_min, y_min, x_max, y_max = boundingbox
    return (x_max - x_min) * (y_max - y_min)


def identity(x, description: Optional[str] = None):
    return x


def euclidean_distance(pos1, pos2):
    return math.sqrt(
        (float(pos1[0]) - float(pos2[0])) ** 2 + (float(pos1[2]) - float(pos2[1])) ** 2
    )


def calculate_relative_angle(coord1, coord2):
    """
    Calculates the relative angle of coord1 from coord2.

    Parameters:
    coord1 (tuple): The (x, y, z) coordinates of the first point.
    coord2 (tuple): The (x, z) coordinates of the second point.

    Returns:
    float: The relative angle in degrees from coord2 to coord1.
    """
    delta_x = float(coord1[0]) - float(coord2[0])
    delta_y = float(coord1[2]) - float(coord2[1])

    angle_degrees = math.degrees(math.pi / 2 - math.atan2(delta_y, delta_x))

    # Ensure the angle is between 0 and 360 degrees
    if angle_degrees < 0:
        angle_degrees += 360

    return angle_degrees


def calculate_surrounding_angles(center_angle):
    """
    Calculates the center angle, left 90 degrees angle, and right 90 degrees angle.

    Parameters:
    center_angle (float): The center angle in degrees.

    Returns:
    tuple: A tuple containing the center angle, left 90 degrees angle, and right 90 degrees angle.
    """
    # Calculate left 90 degrees angle
    left_angle = (center_angle + 90) % 360
    # Calculate right 90 degrees angle
    right_angle = (center_angle - 90) % 360

    # Ensure angles are positive
    if left_angle < 0:
        left_angle += 360
    if right_angle < 0:
        right_angle += 360

    return center_angle, left_angle, right_angle


def is_angle_between(angle, left_angle, right_angle):
    """
    Checks if an angle is between two other angles, accounting for circular nature of angles.

    Parameters:
    angle (float): The angle to check.
    left_angle (float): The left boundary angle.
    right_angle (float): The right boundary angle.

    Returns:
    bool: True if the angle is between left_angle and right_angle, False otherwise.
    """
    if left_angle < right_angle:
        return angle <= left_angle or angle >= right_angle

    else:
        return right_angle <= angle <= left_angle
    # if left_angle < right_angle:
    #     return right_angle <= angle <= left_angle

    # else:
    #     return angle <= left_angle or angle >= right_angle


def filter_nodes_by_distance(
    graph_nodes: list, positions: list, max_distance: float = 2.0
) -> list:
    """
    Filters nodes in the graph G that are within a specified distance from a given position.

    Parameters:
    graph_nodes (list): A list of nodes in the graph.
    position (tuple): The (x, y) position from which distances are calculated.
    max_distance (float): The maximum distance to filter nodes.

    Returns:
    list: A list of nodes within the specified distance from the given position.
    """
    filtered_nodes = set()

    for node in graph_nodes:
        for position in positions:
            distance = euclidean_distance(position, node.split("|"))
            if distance <= max_distance:
                relative_angle = calculate_relative_angle(position, node.split("|"))
                _, left_relative_angle, right_relative_angle = (
                    calculate_surrounding_angles(relative_angle)
                )
                if is_angle_between(
                    int(node.split("|")[2]), left_relative_angle, right_relative_angle
                ):
                    filtered_nodes.add(node)
                    break

    return list(filtered_nodes)


class iTHORScene:
    def __init__(self, controller: Controller, **kwargs):
        self.controller = controller

        self.rotations = (
            [0, 90, 180, 270, 45, 135, 225, 315]
            if "rotations" not in kwargs
            else kwargs["rotations"]
        )
        self.horizons = [-30, 0, 30] if "horizons" not in kwargs else kwargs["horizons"]
        self.boundingbox_size_threshold = (
            100
            if "boundingbox_size_threshold" not in kwargs
            else kwargs["boundingbox_size_threshold"]
        )

    @classmethod
    def generate_scene_graph(cls, controller: Controller) -> nx.DiGraph:
        ithor_scene = cls(controller)

        graph = nx.DiGraph()

        event = ithor_scene.controller.step(dict(action="GetReachablePositions"))
        reachable_positions = event.metadata["actionReturn"]
        reachable_position_list = [[pos["x"], pos["z"]] for pos in reachable_positions]

        for pos in reachable_positions:
            for rotation in ithor_scene.rotations:
                for i_horizon, horizon in enumerate(ithor_scene.horizons):
                    pos_str = "{:0.2f}|{:0.2f}|{:d}|{:d}".format(
                        pos["x"], pos["z"], round(rotation), round(horizon)
                    )

                    # Move Ahead
                    if rotation == 0:
                        moved_x = pos["x"]
                        moved_z = pos["z"] + 0.25
                    elif rotation == 90:
                        moved_x = pos["x"] + 0.25
                        moved_z = pos["z"]
                    elif rotation == 180:
                        moved_x = pos["x"]
                        moved_z = pos["z"] - 0.25
                    elif rotation == 270:
                        moved_x = pos["x"] - 0.25
                        moved_z = pos["z"]
                    elif rotation == 45:
                        moved_x = pos["x"] + 0.25
                        moved_z = pos["z"] + 0.25
                    elif rotation == 135:
                        moved_x = pos["x"] + 0.25
                        moved_z = pos["z"] - 0.25
                    elif rotation == 225:
                        moved_x = pos["x"] - 0.25
                        moved_z = pos["z"] - 0.25
                    elif rotation == 315:
                        moved_x = pos["x"] - 0.25
                        moved_z = pos["z"] + 0.25

                    if [moved_x, moved_z] in reachable_position_list:
                        moved_pos_str = "{:0.2f}|{:0.2f}|{:d}|{:d}".format(
                            moved_x, moved_z, round(rotation), round(horizon)
                        )
                        graph.add_edge(pos_str, moved_pos_str)

                    # Rotate Right
                    moved_pos_str = "{:0.2f}|{:0.2f}|{:d}|{:d}".format(
                        pos["x"], pos["z"], round((rotation + 45) % 360), round(horizon)
                    )
                    graph.add_edge(pos_str, moved_pos_str)

                    # Rotate Left
                    moved_pos_str = "{:0.2f}|{:0.2f}|{:d}|{:d}".format(
                        pos["x"], pos["z"], round((rotation - 45) % 360), round(horizon)
                    )
                    graph.add_edge(pos_str, moved_pos_str)

                    # Look Up/Down
                    if i_horizon > 0 and i_horizon < len(ithor_scene.horizons) - 1:
                        moved_pos_str = "{:0.2f}|{:0.2f}|{:d}|{:d}".format(
                            pos["x"],
                            pos["z"],
                            round(rotation),
                            round(ithor_scene.horizons[(i_horizon + 1)]),
                        )
                        graph.add_edge(pos_str, moved_pos_str)
                        moved_pos_str = "{:0.2f}|{:0.2f}|{:d}|{:d}".format(
                            pos["x"],
                            pos["z"],
                            round(rotation),
                            round(ithor_scene.horizons[(i_horizon - 1)]),
                        )
                        graph.add_edge(pos_str, moved_pos_str)
                    elif i_horizon == 0:
                        moved_pos_str = "{:0.2f}|{:0.2f}|{:d}|{:d}".format(
                            pos["x"],
                            pos["z"],
                            round(rotation),
                            round(ithor_scene.horizons[(i_horizon + 1)]),
                        )
                        graph.add_edge(pos_str, moved_pos_str)
                    elif i_horizon == len(ithor_scene.horizons) - 1:
                        moved_pos_str = "{:0.2f}|{:0.2f}|{:d}|{:d}".format(
                            pos["x"],
                            pos["z"],
                            round(rotation),
                            round(ithor_scene.horizons[(i_horizon - 1)]),
                        )
                        graph.add_edge(pos_str, moved_pos_str)
                    else:
                        raise ValueError("Invalid horizon index")

        return graph

    def get_init_position(self) -> dict:
        return {
            "position": self.controller.last_event.metadata["agent"]["position"],
            "rotation": self.controller.last_event.metadata["agent"]["rotation"],
            "horizon": self.controller.last_event.metadata["agent"]["cameraHorizon"],
        }

    @classmethod
    def get_object_visibility_map(
        cls,
        controller: Controller,
        scene_graph: nx.DiGraph,
        instance_ids: Optional[list] = None,
        progress_function: Any = identity,
        progress_description: Optional[str] = None,
        force_action: bool = False,
        filter_max_distance: Optional[float] = 1.5,
    ) -> tuple[dict, dict]:
        ithor_scene = cls(controller)
        init_position = ithor_scene.get_init_position()

        object_visibility_map = {}
        if instance_ids is None or filter_max_distance == None:
            iter_nodes = scene_graph.nodes()
            if instance_ids is None:
                instance_ids = []
                instance_list = ithor_scene.controller.last_event.metadata["objects"]

                for instance_info in instance_list:
                    instance_id = instance_info["objectId"]
                    instance_ids.append(instance_id)
        else:
            iter_nodes = filter_nodes_by_distance(
                scene_graph.nodes(),
                [item.split("|")[1:] for item in instance_ids],
                max_distance=filter_max_distance,
            )

        for instance_id in instance_ids:
            object_visibility_map[instance_id] = []

        object_visibility_size = {}
        for node in progress_function(iter_nodes, progress_description):
            agent_location = iTHORPosition(node)

            event = ithor_scene.controller.step(
                action="Teleport",
                standing=True,
                forceAction=force_action,
                **agent_location.to_dict(),
            )
            assert ithor_scene.controller.initialization_parameters[
                "renderInstanceSegmentation"
            ], "renderInstanceSegmentation must be True"
            instance_detections = (
                ithor_scene.controller.last_event.instance_detections2D
            )

            for instance in ithor_scene.controller.last_event.metadata["objects"]:
                if (
                    instance["objectId"] in object_visibility_map
                    and instance["visible"]
                    and instance["objectId"] in instance_detections
                    and (
                        calculate_boundingbox_size(
                            instance_detections[instance["objectId"]]
                        )
                        > ithor_scene.boundingbox_size_threshold
                    )
                ):
                    object_visibility_map[instance["objectId"]].append(node)
                    if node not in object_visibility_size:
                        object_visibility_size[node] = {}
                    object_visibility_size[node][instance["objectId"]] = (
                        calculate_boundingbox_size(
                            instance_detections[instance["objectId"]]
                        )
                    )

        event = controller.step(
            action="Teleport",
            position=init_position["position"],
            rotation=init_position["rotation"],
            horizon=init_position["horizon"],
            standing=True,
            forceAction=force_action,
        )

        return object_visibility_map, object_visibility_size

    @classmethod
    def filter_potential_targets(
        cls,
        environment,
        scene_graph: nx.DiGraph,
        n_max_targets: Optional[int] = None,
    ):
        """
        Filters potential target objects in an environment based on visibility and other criteria.

        This method performs the following steps:
        1. Retrieve Target IDs: Fetches a list of potential target IDs from the environment, with an optional limit.
        2. Compute Object Visibility: Calculates the visibility of objects using the get_object_visibility_map method.
        3. Filter Out Invisible Targets: Removes target IDs that are not visible from the list.
        4. Class and Instance ID Filtering: Attempts to find replacements for removed targets by filtering based on class and instance ID, then randomly selecting new targets.
        5. Remove Targets Not in Visibility Map: Removes any target IDs not in the object visibility map.
        6. Find New Targets for Removed IDs: For each removed target ID, repeats the process of finding a replacement target.
        7. Return Values: Returns the updated list of target IDs, the object visibility map, and the object visibility size metric.

        Parameters:
        - environment (Environment): The environment to filter targets from.
        - scene_graph (nx.DiGraph): The scene graph of the environment.
        - n_max_targets (Optional[int]): The maximum number of target IDs to retrieve.

        Returns:
        - tuple: A tuple containing the updated list of target IDs, the object visibility map, and the object visibility size metric.
        """
        target_ids = environment.get_target_ids(n_max_targets=n_max_targets)
        object_visibility_map, object_visibility_size = (
            iTHORScene.get_object_visibility_map(
                environment.controller, scene_graph, force_action=True
            )
        )

        for object_id in object_visibility_map:
            if len(object_visibility_map[object_id]) < 1 and object_id in target_ids:
                target_ids.remove(object_id)

                _target_class = sorted(
                    list(set([item.split("|")[0] for item in target_ids]))
                )
                acceptable_instance_ids = [
                    item
                    for item in object_visibility_map
                    if item not in object_id and len(object_visibility_map[item]) > 0
                ]

                _instance_ids = [
                    item
                    for item in acceptable_instance_ids
                    if item.split("|")[0] not in _target_class
                ]
                if len(_instance_ids) > 0:
                    acceptable_instance_ids = _instance_ids

                target_ids.append(random.choice(acceptable_instance_ids))

        remove_target_ids = []
        for target_id in target_ids:
            if target_id not in object_visibility_map:
                remove_target_ids.append(target_id)

        # find a new target instead
        for remove_target_id in remove_target_ids:
            target_ids.remove(remove_target_id)

            _target_class = sorted(
                list(set([item.split("|")[0] for item in target_ids]))
            )
            acceptable_instance_ids = [
                item
                for item in object_visibility_map
                if item not in target_ids and len(object_visibility_map[item]) > 0
            ]

            _instance_ids = [
                item
                for item in acceptable_instance_ids
                if item.split("|")[0] not in _target_class
            ]
            if len(_instance_ids) > 0:
                acceptable_instance_ids = _instance_ids

            target_ids.append(random.choice(acceptable_instance_ids))

        return target_ids, object_visibility_map, object_visibility_size

    @classmethod
    def calculate_boundingbox_size(cls, boundingbox: list) -> float:
        x_min, y_min, x_max, y_max = boundingbox
        return (x_max - x_min) * (y_max - y_min)


def map_to_nearest_rotation_angle(value):
    angles = [0, 45, 90, 135, 180, 225, 270, 315]
    closest_angle = min(angles, key=lambda x: abs(x - value))
    return closest_angle


def map_to_nearest_horizon_angle(value):
    angles = [-30, 0, 30]
    closest_angle = min(angles, key=lambda x: abs(x - value))
    return closest_angle


def map_to_nearest_grid_point(value, grid_size=0.25):
    grid_points = np.arange(-12, 12, grid_size, dtype=float)
    closest_grid_point = min(grid_points, key=lambda x: abs(x - value))
    return closest_grid_point


@dataclass
class iTHORPosition:
    input: Union[str, Controller, dict] = field(default_factory=str)

    position_x: float = field(default_factory=float)
    position_y: float = field(default=0.900999128818512)
    position_z: float = field(default_factory=float)

    rotation_y: float = field(default_factory=int)
    horizon: float = field(default_factory=int)

    def __post_init__(self):
        if isinstance(self.input, str) and "|" in self.input:
            position_x, position_z, rotation_y, horizon = self.input.split("|")
            self.position_x = map_to_nearest_grid_point(float(position_x))
            self.position_z = map_to_nearest_grid_point(float(position_z))
            self.rotation_y = int(rotation_y)
            self.horizon = int(horizon)

        elif isinstance(self.input, Controller):
            position_x = self.input.last_event.metadata["agent"]["position"]["x"]
            position_z = self.input.last_event.metadata["agent"]["position"]["z"]

            self.position_x = map_to_nearest_grid_point(float(position_x))
            self.position_z = map_to_nearest_grid_point(float(position_z))

            rotation = self.input.last_event.metadata["agent"]["rotation"]
            self.rotation_y = int(map_to_nearest_rotation_angle(rotation["y"]))

            horizon = self.input.last_event.metadata["agent"]["cameraHorizon"]
            self.horizon = int(map_to_nearest_horizon_angle(horizon))

        elif isinstance(self.input, dict):
            if "x" in self.input:
                self.position_x = map_to_nearest_grid_point(self.input["x"])
            if "y" in self.input:
                self.position_y = self.input["y"]
            if "z" in self.input:
                self.position_z = map_to_nearest_grid_point(self.input["z"])
            if "rotation" in self.input:
                if isinstance(self.input["rotation"], dict):
                    self.rotation_y = self.input["rotation"]["y"]
                elif isinstance(
                    self.input["rotation"], (float, np.floating)
                ) or isinstance(self.input["rotation"], (int, np.integer)):
                    self.rotation_y = int(self.input["rotation"])
                else:
                    self.rotation_y = random.choice(np.arange(0, 360, 45))
            else:
                self.rotation_y = random.choice(np.arange(0, 360, 45))
            if "horizon" in self.input:
                self.horizon = self.input["horizon"]
            else:
                self.horizon = random.choice([0, 30, -30])

        elif isinstance(self.input, (h5py.File, h5py.Group, h5py.Dataset)):
            position_x = json.loads(self.input["position"]["x"][()])
            position_z = json.loads(self.input["position"]["z"][()])

            self.position_x = map_to_nearest_grid_point(float(position_x))
            self.position_z = map_to_nearest_grid_point(float(position_z))

            self.rotation_y = int(
                map_to_nearest_rotation_angle(
                    json.loads(self.input["rotation"]["y"][()])
                )
            )

            self.horizon = int(
                map_to_nearest_horizon_angle(
                    json.loads(self.input["cameraHorizon"][()])
                )
            )

        else:
            raise NotImplementedError(f"Invalid input type: {type(self.input)}")

    def __repr__(self) -> str:
        return "{:0.2f}|{:0.2f}|{:d}|{:d}".format(
            self.position_x,
            self.position_z,
            round(self.rotation_y),
            round(self.horizon),
        )

    def to_dict(self):
        return {
            "position": {
                "x": self.position_x,
                "y": self.position_y,
                "z": self.position_z,
            },
            "rotation": {"x": 0.0, "y": self.rotation_y, "z": 0.0},
            "horizon": self.horizon,
        }


if __name__ == "__main__":
    from ai2thor.platform import CloudRendering
    from rich.progress import track

    controller = Controller(
        platform=CloudRendering,
        agentMode="default",
        visibilityDistance=1.5,
        scene="FloorPlan212",
        # step sizes
        gridSize=0.25,
        snapToGrid=True,
        rotateStepDegrees=90,
        # image modalities
        renderDepthImage=False,
        renderInstanceSegmentation=True,
        # camera properties
        width=300,
        height=300,
        fieldOfView=90,
    )

    print(str(iTHORPosition("-0.00|-0.00|270|0")))
    assert str(iTHORPosition("-0.00|-0.00|270|0")) == "0.00|0.00|270|0", "Failed"

    event = controller.step(
        action="GetInteractablePoses",
        objectId="ArmChair|-00.27|+00.00|+01.87",
        rotations=np.linspace(0, 360, 9),
        horizons=np.linspace(-30, 30, 3),
        standings=[True],
    )

    poses = event.metadata["actionReturn"]

    scene_graph = iTHORScene.generate_scene_graph(controller)

    target_ids = []
    instance_list = controller.last_event.metadata["objects"]
    for instance_info in instance_list:
        target_id = instance_info["objectId"]
        target_ids.append(target_id)

    iTHORScene.get_object_visibility_map(
        controller,
        scene_graph,
        target_ids,
        progress_function=track,
        progress_description="Building visibility map",
    )

    # target_ids = random.sample(target_ids, k=10)

    # graph_data = nx.readwrite.json_graph.node_link_data(scene_graph)
    # with open("scene_graph.json", "w") as f:
    #     json.dump(graph_data, f)

    controller.stop()
