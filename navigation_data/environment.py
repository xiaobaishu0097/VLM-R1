import copy
import math
import random
import re
from typing import Any, Optional, Tuple, Union

import inflect
import networkx as nx
import numpy as np
from ai2thor.controller import Controller
from scene import iTHORPosition
from utils.detection_utils import (
    calculate_boundingbox_center,
    calculate_boundingbox_size,
)
from utils.ithor_utils import calculate_instance_distance
from utils.logger import Logger
from utils.text_utils import (
    concat_list,
    get_indefinite_article,
    remove_last_comma,
    replace_last_comma,
    split_string_at_capital,
)
from utils.visualization_utils import identity

TARGET_CLASS_FILTER = ["Floor", "Ceiling", "Wall"]


class Environment:
    def __init__(
        self,
        configs: dict,
        scene: Optional[str] = None,
        target_class: str = "ArmChair",
        # Sensor parameters
        always_include_observed_instances: bool = False,
        **kwargs,
    ) -> None:
        self.logger = Logger.get_logger(name=configs["experiment_id"])
        self.environment_id = (
            configs["environment_id"] if "environment_id" in configs else 0
        )

        self._scene = scene
        self._target_class = target_class

        self.observation_width = 300
        self.observation_height = 300
        self.boundingbox_size_threshold = 100

        # TODO: this should be set in the Agent class instead of the Environment class
        self.always_include_observed_instances = always_include_observed_instances

    @property
    def controller(self) -> Controller:
        raise NotImplementedError

    def initialize_controller(self, scene: str) -> Controller:
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    def random_reset(self, scene: str) -> None:
        raise NotImplementedError

    @property
    def scene(self) -> str:
        if not hasattr(self, "_scene"):
            raise ValueError("Scene is not set.")
        return self._scene

    @property
    def scene_id(self) -> str:
        raise NotImplementedError

    @property
    def observation(self) -> np.ndarray:
        raise NotImplementedError

    @property
    def position(self) -> iTHORPosition:
        raise NotImplementedError

    @property
    def step_metadata(self) -> dict:
        raise NotImplementedError

    @property
    def step_agent_metadata(self):
        raise NotImplementedError

    @property
    def step_agent_position_metadata(self) -> dict:
        raise NotImplementedError

    @property
    def step_agent_rotation_metadata(self) -> dict:
        raise NotImplementedError

    @property
    def step_agent_cameraHorizon_metadata(self) -> int:
        raise NotImplementedError

    @property
    def step_object_metadata(self) -> list[dict]:
        raise NotImplementedError

    @property
    def observation_detection(self) -> dict:
        raise NotImplementedError

    def get_object_visibility_map(
        self,
        progress_function: Any = identity,
        progress_description: Optional[str] = None,
    ) -> dict:
        raise NotImplementedError

    def teleport_agent(self, position: iTHORPosition) -> None:
        raise NotImplementedError

    @property
    def target_bounding_box(self) -> Union[list, np.ndarray]:
        return self.observation_detection[self.target_id]

    @property
    def target_class(self) -> str:
        if not hasattr(self, "_target_class"):
            raise ValueError("Target class not set")
        return self._target_class

    @property
    def potential_target_ids(self) -> list[str]:
        if not hasattr(self, "_potential_target_ids"):
            raise ValueError("Potential target IDs not set")
        return self._potential_target_ids

    def reset_target(self, target_class: str) -> None:
        self._target_class = target_class
        _, self._potential_target_ids = self.check_target_class(target_class)
        if len(self._potential_target_ids) > 0:
            if hasattr(self, "_target_id"):
                del self._target_id
        else:
            self._target_id = self.closest_target_id

        if hasattr(self, "target_direction"):
            del self.target_direction
            del self.last_succeeded_action
        if hasattr(self, "target_is_observed"):
            del self.target_is_observed
        if hasattr(self, "_closest_target_id"):
            del self._closest_target_id

    def get_observsed_object_classes(self) -> list[str]:
        observed_object_classes = []

        for object_id in self.observation_detection.keys():
            object_class = object_id.split("|")[0]
            if (
                ":" not in object_id
                and object_class not in observed_object_classes
                and object_class not in TARGET_CLASS_FILTER
            ):
                observed_object_classes.append(object_class)

        return observed_object_classes

    def get_target_class_visible_positions(self, target_class: str) -> list:
        visible_positions = []
        for target_id in self.object_visibility_map:
            if target_id.split("|")[0] == target_class:
                visible_positions += self.object_visibility_map[target_id]
        visible_positions = list(set(visible_positions))
        return visible_positions

    def get_target_ids(self, n_max_targets: Optional[int] = None) -> list:
        target_ids = []
        instance_list = self.step_object_metadata

        for instance_info in instance_list:
            if (
                instance_info["pickupable"]
                or instance_info["openable"]
                or instance_info["receptacle"]
                or instance_info["toggleable"]
                or instance_info["breakable"]
                or instance_info["moveable"]
            ) and instance_info["objectType"] != "Floor":
                target_ids.append(instance_info["objectId"])

        if n_max_targets is not None and n_max_targets < len(target_ids):
            target_ids = random.sample(target_ids, k=n_max_targets)

        return target_ids

    def get_all_object_classes(self) -> list[str]:
        all_object_classes = []

        for object_info in self.step_object_metadata:
            object_class = object_info["objectType"]
            if (
                object_class not in all_object_classes
                and object_class not in TARGET_CLASS_FILTER
            ):
                all_object_classes.append(object_class)

        return all_object_classes

    def filter_object_ids(self, object_ids: list[str]) -> list[str]:
        filtered_object_ids = []
        for object_id in object_ids:
            if object_id.split("|")[0] not in TARGET_CLASS_FILTER:
                filtered_object_ids.append(object_id)
        return filtered_object_ids

    def check_target_class(self, target_class: str) -> Tuple[bool, list[str]]:
        """Check if the target class exists in the environment and return the target IDs if it does."""
        target_existing = False
        target_ids = []

        for object_info in self.step_object_metadata:
            if target_class.lower() == object_info["objectType"].lower():
                target_existing = True
                target_ids.append(object_info["objectId"])

        # Temporary assert statement to ensure the target class exists in the environment.
        # TODO: Implement a more robust error handling mechanism.
        assert (
            target_existing and len(target_ids) > 0
        ), f"Target class {target_class} not found in the environment."

        return target_existing, target_ids

    def get_positions(
        self, n_max_positions: Optional[int] = None, detailed_position: bool = False
    ) -> list[iTHORPosition]:
        raise NotImplementedError

    def is_valid_episode(self, navigation_episode: list, target_class: str) -> bool:
        raise NotImplementedError

    def set_target_id(self, target_id: str) -> None:
        self._target_id = target_id
        self._target_class = target_id.split("|")[0]

    @property
    def target_id(self) -> str:
        if not hasattr(self, "_target_id"):
            return self.closest_target_id
        return self._target_id

    def get_optimal_path_length(self) -> int:
        optimal_path_length = len(self.shortest_path)
        return optimal_path_length

    @property
    def optimal_action(self) -> str:
        optimal_action = self.get_action_to_target(
            self.target_id, graph=self.scene_graph
        )
        return optimal_action

    def get_action_to_target(
        self, target_id: str, graph: nx.DiGraph, navigation_path: Optional[list] = None
    ) -> str:
        if navigation_path is None:
            object_visible_positions = self.object_visibility_map[target_id]
            assert (
                str(self.position) in graph.nodes
            ), f"Current position {str(self.position)} not in the graph."
            shortest_path = self.estimate_shortest_path(
                str(self.position), graph, object_visible_positions
            )
        else:
            shortest_path = navigation_path
        optimal_action = self.get_optimal_action(self.position.to_dict(), shortest_path)

        return optimal_action

    @property
    def shortest_path(self):
        object_visible_positions = self.object_visibility_map[self.target_id]
        assert (
            str(self.position) in self.scene_graph.nodes
        ), f"Current position {str(self.position)} not in the graph."
        shortest_path = self.estimate_shortest_path(
            str(self.position),
            self.scene_graph,
            object_visible_positions,
        )

        return shortest_path

    def estimate_shortest_path(
        self,
        current_state: str,
        graph: nx.DiGraph,
        object_visible_positions: list,
    ) -> list:
        shortest_path = []

        for object_visible_position in object_visible_positions:
            try:
                short_path = nx.shortest_path(
                    graph, current_state, object_visible_position
                )
            except Exception:
                short_path = []

            if len(short_path) > 0 and (
                (len(shortest_path) == 0) or (len(short_path) < len(shortest_path))
            ):
                shortest_path: list = short_path

        return shortest_path

    def get_optimal_action(self, current_state: dict, optimal_path: list) -> str:
        action_prob = "MoveAhead"
        if len(optimal_path) <= 1:
            action_prob = "Done"
        else:
            next_state = optimal_path[1]

            current_x, current_z, current_rot, current_hor = (
                current_state["position"]["x"],
                current_state["position"]["z"],
                current_state["rotation"]["y"],
                current_state["horizon"],
            )
            next_state = iTHORPosition(next_state).to_dict()
            next_x, next_z, next_rot, next_hor = (
                next_state["position"]["x"],
                next_state["position"]["z"],
                next_state["rotation"]["y"],
                next_state["horizon"],
            )

            if int(next_rot) == (int(current_rot) + 45) % 360:
                action_prob = "RotateRight"
            elif int(next_rot) == (int(current_rot) - 45) % 360:
                action_prob = "RotateLeft"
            elif (current_hor == 0 and next_hor == -30) or (
                current_hor == 30 and next_hor == 0
            ):
                action_prob = "LookUp"
            elif (current_hor == -30 and next_hor == 0) or (
                current_hor == 0 and next_hor == 30
            ):
                action_prob = "LookDown"
            elif (current_x != next_x) or (current_z != next_z):
                action_prob = "MoveAhead"
            else:
                raise ValueError(
                    f"Unknown action, from {current_state} to {next_state}"
                )

        return action_prob

    def get_visible_target_ids(self) -> list[str]:
        visible_target_ids = []

        for potential_target_id in self._potential_target_ids:
            if potential_target_id in self.observation_detection.keys():
                if (
                    calculate_boundingbox_size(
                        self.observation_detection[potential_target_id]
                    )
                    > self.boundingbox_size_threshold
                ):
                    visible_target_ids.append(potential_target_id)

        return visible_target_ids

    def check_instance_visibility(self, target_id: str) -> bool:
        instance_detection = self.observation_detection
        if target_id in instance_detection:
            if (
                calculate_boundingbox_size(instance_detection[target_id])
                > self.boundingbox_size_threshold
            ):
                return True
        return False

    def check_target_visibility(self) -> bool:
        assert (
            self.target_id.split("|")[0] == self.target_class
        ), f"Target class mismatch: {self.target_id} vs {self.target_class}"
        if self.target_id in self.observation_detection.keys():
            return True
        return False

    @property
    def scene_graph(self):
        if not hasattr(self, "_scene_graph"):
            raise ValueError("Scene graph not set")
        return self._scene_graph

    def set_scene_graph(self, scene_graph: nx.DiGraph):
        self._scene_graph = scene_graph

    @property
    def object_visibility_map(self):
        if not hasattr(self, "_object_visibility_map"):
            raise ValueError("Object visibility map not set")
        return self._object_visibility_map

    def get_target_position(self) -> dict:
        for object_info in self.step_object_metadata:
            if object_info["objectId"] == self.target_id:
                return object_info["position"]
        return {"x": 0, "y": 0, "z": 0}

    def is_left_or_right(
        self, target_position: dict, agent_position: dict, rotation: int
    ):
        target_vector_x = target_position["x"] - agent_position["x"]
        target_vector_y = target_position["z"] - agent_position["z"]

        rotation_rad = math.radians(rotation)
        robot_forward_vector_x = math.cos(rotation_rad)
        robot_forward_vector_y = math.sin(rotation_rad)

        cross_product = (robot_forward_vector_x * target_vector_y) - (
            robot_forward_vector_y * target_vector_x
        )

        if cross_product > 0:
            return "left"
        elif cross_product < 0:
            return "right"
        else:
            return "front"

    def get_target_relative_position(self) -> dict:
        target_position = self.get_target_position()
        agent_position = self.step_agent_position_metadata

        relative_position = self.is_left_or_right(
            target_position=target_position,
            agent_position=agent_position,
            rotation=self.step_agent_rotation_metadata["y"],
        )

        return {
            "agent_position": agent_position,
            "target_position": target_position,
            "relative_position": relative_position,
        }

    @property
    def target_relative_position(self) -> str:
        return self.get_target_relative_position()["relative_position"]

    @property
    def observation_description(self) -> str:
        if self.check_instance_visibility(self.target_id):
            # observation_description: str = (
            #     f"If the target is {self.target_distance_description} me and is situated in the {self.target_position_description} of the observational field."
            # )
            if not self.always_include_observed_instances:
                observation_description = (
                    f"At the current position, the target ({self.target_class}) is"
                    f" {self.target_distance_description} you and is situated in the"
                    f" {self.target_position_description} of the observational field."
                )
            else:
                observation_description = (
                    f"At the current position, the target ({self.target_class}) is"
                    f" {self.target_distance_description} you and is situated in the"
                    f" {self.target_position_description} of the observational field."
                    f" {self.observed_instance_positional_description}"
                )

        else:
            if hasattr(self, "target_direction"):
                # observation_description = (
                #     f"The target might be on the {self.target_direction} side of me."
                # )
                if not self.always_include_observed_instances:
                    observation_description = (
                        "At the current position, you cannot observe the target"
                        f" ({self.target_class}). However, based on the history, the"
                        f" target might be on the {self.target_relative_position} side"
                        " of you."
                    )
                else:
                    observation_description = (
                        "At the current position, you cannot observe the target"
                        f" ({self.target_class}). However, based on the history, the"
                        f" target might be on the {self.target_relative_position} side"
                        " of you."
                        f" {self.observed_instance_positional_description}"
                    )

            else:
                observation_description = (
                    "At the current position, you cannot observe the target"
                    f" ({self.target_class})."
                    f" {self.observed_instance_positional_description}"
                )

        return observation_description

    @property
    def observed_instance_positional_description(self) -> str:
        instance_detections2D = self.observation_detection
        instance_ids = self.extract_meaningful_instance_ids(instance_detections2D)

        visible_areas = {"left": [], "center": [], "right": []}
        for instance_id in instance_ids:
            if instance_id in instance_detections2D:
                instance_area = self.get_instance_position_description(
                    instance_id, n_areas=3
                )
                visible_areas[instance_area].append(instance_id)

        area_description = None
        brief_description = []
        for area in visible_areas:
            if len(visible_areas[area]) > 0:
                if area_description is None:
                    area_description = "You can observe "
                else:
                    area_description = ""

                area_instance_counter = {}
                for instance_id in visible_areas[area]:
                    if instance_id.split("|")[0] not in area_instance_counter:
                        area_instance_counter[instance_id.split("|")[0]] = 0
                    area_instance_counter[instance_id.split("|")[0]] += 1

                for instance_category in area_instance_counter:
                    if area_instance_counter[instance_category] > 1:
                        area_description += (
                            f"{area_instance_counter[instance_category]} {split_string_at_capital(instance_category).lower()}s, "
                        )
                    else:
                        area_description += (
                            f"{get_indefinite_article(split_string_at_capital(instance_category).lower())}, "
                        )

                area_description = area_description.rsplit(", ", 1)[0] + " "
                if area == "left" or area == "right":
                    area_description += "at the " + area + " side"
                else:
                    area_description += "at the " + area

                brief_description.append(area_description)

        if len(brief_description) == 0:
            brief_description = "You cannot observe anything."
        else:
            brief_description = remove_last_comma(concat_list(brief_description)) + "."

        return brief_description

    def extract_meaningful_instance_ids(self, instance_detections2D: dict) -> list[str]:
        instance_ids = []
        for instance_info in self.step_object_metadata:
            if (
                instance_info["pickupable"]
                or instance_info["openable"]
                or instance_info["receptacle"]
                or instance_info["toggleable"]
                or instance_info["breakable"]
                or instance_info["moveable"]
            ) and instance_info["objectType"] != "Floor":
                instance_ids.append(instance_info["objectId"])

        extra_instance_ids = [
            item
            for item in instance_detections2D.keys()
            if "Door" == item.split("|")[0]
        ]
        if len(extra_instance_ids) > 0:
            instance_ids += extra_instance_ids
        return instance_ids

    def get_surrounding_instances(self, target_id: str) -> list:
        target_x = float(target_id.split("|")[1])
        target_z = float(target_id.split("|")[3])
        target_info = {"x": target_x, "z": target_z}

        surrounding_instances = []
        for instance_info in self.step_object_metadata:
            if (
                (
                    instance_info["pickupable"]
                    or instance_info["openable"]
                    or instance_info["receptacle"]
                    or instance_info["toggleable"]
                    or instance_info["breakable"]
                    or instance_info["moveable"]
                )
                and instance_info["objectType"] != "Floor"
                and instance_info["objectId"] != target_id
                and (
                    calculate_instance_distance(target_info, instance_info["position"])
                    < self.surrounding_distance_threshold
                )
            ):
                surrounding_instances.append(instance_info["objectId"])

        return surrounding_instances

    @property
    def surrounding_description(self) -> str:
        return self.get_surrounding_description(self.target_id)

    def get_surrounding_description(self, target_id: str) -> str:
        surrounding_instances = self.get_surrounding_instances(target_id)

        receptacle_ids = self.get_receptacle(target_id)
        if len(receptacle_ids) > 0:
            receptacle_id = receptacle_ids[0]
            if receptacle_id in surrounding_instances:
                surrounding_instances.remove(receptacle_id)

        if len(surrounding_instances) > 0:
            surrounding_instance_categories = [
                item.split("|")[0] for item in surrounding_instances
            ]

            p = inflect.engine()
            surrounding_instance_counter = {}
            surrounding_instance_description = ""
            for instance in surrounding_instances:
                if instance.split("|")[0] not in surrounding_instance_counter:
                    if (
                        surrounding_instance_categories.count(instance.split("|")[0])
                        > 1
                    ):
                        surrounding_instance_counter[instance.split("|")[0]] = 1
                        surrounding_instance_description += (
                            f"the {p.ordinal(1)} {split_string_at_capital(instance.split('|')[0]).lower()}, "
                        )
                    else:
                        surrounding_instance_description += (
                            f"{get_indefinite_article(split_string_at_capital(instance.split('|')[0])).lower()}, "
                        )
                else:
                    surrounding_instance_counter[instance.split("|")[0]] += 1
                    surrounding_instance_description += (
                        f"the {p.ordinal(surrounding_instance_counter[instance.split('|')[0]])} {split_string_at_capital(instance.split('|')[0]).lower()}, "
                    )
            surrounding_instance_description = replace_last_comma(
                remove_last_comma(surrounding_instance_description)
            )
            return (
                f" The {split_string_at_capital(target_id.split('|')[0]).lower()} is"
                f" located next to {surrounding_instance_description}."
            )
        else:
            return ""

    def get_receptacle(self, target_id: str) -> list:
        for object_info in self.step_object_metadata:
            if target_id == object_info["objectId"]:
                if object_info["parentReceptacles"] is not None:
                    return object_info["parentReceptacles"]
                else:
                    return []
        return []

    def get_receptacle_position_description(
        self, receptacle_id: str, reachable_positions: Optional[list] = None
    ) -> str:
        receptacle_x = float(receptacle_id.split("|")[1])
        receptacle_z = float(receptacle_id.split("|")[3])

        reachable_positions = self.controller.step(
            action="GetReachablePositions"
        ).metadata["actionReturn"]
        reachable_directions = {
            "lessx": False,
            "morex": False,
            "lessz": False,
            "morez": False,
        }
        assert (
            reachable_positions is not None
        ), f"reachable_positions: {reachable_positions}"
        for reachable_position in reachable_positions:
            if reachable_position["x"] < receptacle_x:
                reachable_directions["lessx"] = True
            elif reachable_position["x"] > receptacle_x:
                reachable_directions["morex"] = True
            if reachable_position["z"] < receptacle_z:
                reachable_directions["lessz"] = True
            elif reachable_position["z"] > receptacle_z:
                reachable_directions["morez"] = True

        if (
            reachable_directions["lessx"]
            and reachable_directions["morex"]
            and reachable_directions["lessz"]
            and reachable_directions["morez"]
        ):
            return "in the center of the scene"
        else:
            return "on the side of the scene"

    def get_receptacle_description(self, target_id: str) -> str:
        receptacle_ids = self.get_receptacle(target_id)
        if len(receptacle_ids) > 0:
            receptacle_id = receptacle_ids[0]

            return (
                " You have knew that the"
                f" {split_string_at_capital(target_id.split('|')[0]).lower()} is on the"
                f" {split_string_at_capital(receptacle_id.split('|')[0]).lower()} that"
                f" is {self.get_receptacle_position_description(receptacle_id)}."
            )
        # there is no receptacle
        else:
            return ""

    @property
    def receptacle_description(self) -> str:
        return self.get_receptacle_description(self.target_id)

    def get_closest_target_id(self) -> Tuple[str, float]:
        visible_target_ids = self.get_visible_target_ids()
        # assert (
        #     len(visible_target_ids) > 0
        # ), "No visible targets found in the environment."

        if len(visible_target_ids) > 0:
            assert all([
                item.split("|")[0] == self.target_class for item in visible_target_ids
            ]), f"Target class mismatch: {visible_target_ids} vs {self.target_class}"
            closest_distance = float("inf")
            closest_target_id: str

            for object_info in self.step_object_metadata:
                if object_info["objectId"] in visible_target_ids:
                    distance = object_info["distance"]
                    if distance < closest_distance:
                        closest_distance = distance
                        closest_target_id = object_info["objectId"]

            self.target_is_observed = True
            self._closest_target_id = closest_target_id
            assert self._closest_target_id.split("|")[0] == self.target_class, (
                f"Target class mismatch: {self._closest_target_id} vs"
                f" {self.target_class}"
            )

        else:
            if hasattr(self, "last_succeeded_action") and getattr(
                self, "target_is_observed", False
            ):
                if not hasattr(self, "target_direction"):
                    if self.last_succeeded_action == "RotateRight":
                        self.target_direction = "left"
                    elif self.last_succeeded_action == "RotateLeft":
                        self.target_direction = "right"

                closest_target_id = self._closest_target_id
                closest_distance = float("inf")

                for object_info in self.step_object_metadata:
                    if object_info["objectId"] == closest_target_id:
                        closest_distance = object_info["distance"]

            else:
                closest_distance = float("inf")
                for object_info in self.step_object_metadata:
                    if object_info["objectType"] == self.target_class:
                        distance = object_info["distance"]
                        if distance < closest_distance:
                            closest_distance = distance
                            closest_target_id = object_info["objectId"]

        return closest_target_id, closest_distance

    @property
    def closest_target_id(self) -> str:
        self._closest_target_id, _ = self.get_closest_target_id()

        return self._closest_target_id

    @property
    def target_distance_description(self) -> str:
        """Return a description of the distance of the target from the agent.
        "close to" if the target is visible, "far from" if the target is not visible.
        """
        if (
            not hasattr(self, "target_distance_description_recorded_position")
            or not hasattr(self, "_target_distance_description")
            or self.target_distance_description_recorded_position
            != iTHORPosition(self.step_agent_metadata)
        ):
            self.target_distance_description_recorded_position = iTHORPosition(
                self.step_agent_metadata
            )
            for object_info in self.step_object_metadata:
                if object_info["objectId"] == self.target_id:
                    target_is_visible = object_info["visible"]
                    break

            if target_is_visible:
                self._target_distance_description = "close to"
            else:
                self._target_distance_description = "far from"

        return self._target_distance_description

    @property
    def target_position_description(self) -> str:
        return self.get_instance_position_description(
            instance_id=self.target_id, n_areas=9
        )

    def get_instance_position_description(
        self, instance_id: str, n_areas: int = 9
    ) -> str:
        # The keys are object IDs and the values are [Upper Left 𝑥, Upper Left 𝑦, Lower Right 𝑥, Lower Right 𝑦], where each element is the number of pixels it is from the top left corner of the image.
        instance_detection_bounding_box = self.observation_detection[instance_id]
        instance_center = calculate_boundingbox_center(instance_detection_bounding_box)

        grid_percentage = 5

        if n_areas == 9:
            if (
                instance_center[0] <= self.observation_width / grid_percentage
                and instance_center[1] <= self.observation_height / grid_percentage
            ):
                description = "upper-left corner"
            elif (
                instance_center[0] <= self.observation_width / grid_percentage
                and instance_center[1] > self.observation_height / grid_percentage
                and instance_center[1]
                <= (grid_percentage - 1) * self.observation_height / grid_percentage
            ):
                description = "middle-left grid"
            elif (
                instance_center[0] <= self.observation_width / grid_percentage
                and instance_center[1]
                > (grid_percentage - 1) * self.observation_height / grid_percentage
            ):
                description = "lower-left corner"
            elif (
                instance_center[0] > self.observation_width / grid_percentage
                and instance_center[0]
                <= (grid_percentage - 1) * self.observation_width / grid_percentage
                and instance_center[1] <= self.observation_height / grid_percentage
            ):
                description = "upper-center grid"
            elif (
                instance_center[0] > self.observation_width / grid_percentage
                and instance_center[0]
                <= (grid_percentage - 1) * self.observation_width / grid_percentage
                and instance_center[1] > self.observation_width / grid_percentage
                and instance_center[1]
                <= (grid_percentage - 1) * self.observation_height / grid_percentage
            ):
                description = "middle-center grid"
            elif (
                instance_center[0] > self.observation_width / grid_percentage
                and instance_center[0]
                <= (grid_percentage - 1) * self.observation_width / grid_percentage
                and instance_center[1]
                > (grid_percentage - 1) * self.observation_height / grid_percentage
            ):
                description = "lower-center grid"
            elif (
                instance_center[0]
                > (grid_percentage - 1) * self.observation_width / grid_percentage
                and instance_center[1] <= self.observation_height / grid_percentage
            ):
                description = "upper-right corner"
            elif (
                instance_center[0]
                > (grid_percentage - 1) * self.observation_width / grid_percentage
                and instance_center[1] > self.observation_height / grid_percentage
                and instance_center[1]
                <= (grid_percentage - 1) * self.observation_height / grid_percentage
            ):
                description = "middle-right grid"
            elif (
                instance_center[0]
                > (grid_percentage - 1) * self.observation_width / grid_percentage
                and instance_center[1]
                > (grid_percentage - 1) * self.observation_height / grid_percentage
            ):
                description = "lower-right corner"
            else:
                raise ValueError(f"Invalid position: {instance_center}")

        elif n_areas == 3:
            if instance_center[0] <= self.observation_width / grid_percentage:
                description = "left"
            elif (
                instance_center[0] > self.observation_width / grid_percentage
                and instance_center[0]
                <= (grid_percentage - 1) * self.observation_width / grid_percentage
            ):
                description = "center"
            elif (
                instance_center[0]
                > (grid_percentage - 1) * self.observation_width / grid_percentage
            ):
                description = "right"
            else:
                raise ValueError(f"Invalid position: {instance_center}")

        else:
            raise ValueError(f"Invalid number of areas: {n_areas}")

        return description

    def act(self, action: str) -> bool:
        raise NotImplementedError

    def check_done_status(self) -> bool:
        for object_info in self.step_object_metadata:
            if object_info["objectId"] == self.target_id:
                return object_info["visible"]
        return False

    @property
    def obstacle_description(self) -> str:
        return self.get_obstacle_description()

    def get_obstacle_description(self) -> str:
        obstacle_description = ""
        current_position = copy.deepcopy(self.position)

        self.controller.step(action="MoveAhead")
        if (
            not self.step_metadata["lastActionSuccess"]
            and "blocking" in self.step_metadata["errorMessage"]
        ):
            blocking_string_id = (
                self.step_metadata["errorMessage"].split(" ").index("blocking")
            )
            if blocking_string_id >= 2:
                obstacle = split_string_at_capital(
                    self.step_metadata["errorMessage"].split(" ")[
                        blocking_string_id - 2
                    ]
                )
                obstacle = re.sub(r"\d+", "", obstacle).strip()

                obstacle_description += (
                    f" However, there is an obstacle {obstacle} in front of you."
                )
            else:
                obstacle_description += (
                    f" However, there is an obstacle in front of you."
                )
        else:
            self.controller.step(action="MoveBack")

        if self.position.to_dict() != current_position.to_dict():
            self.controller.step(
                action="Teleport",
                **current_position.to_dict(),
                standing=True,
                forceAction=True,
            )

        return obstacle_description

    @property
    def observed_boundingbox_description(self) -> str:
        return self.get_observed_boundingbox_description()

    def get_observed_boundingbox_description(self) -> str:
        instance_detections2D = self.observation_detection

        visible_instances = self.extract_meaningful_instance_ids(instance_detections2D)
        visible_instances = [
            item for item in visible_instances if item in instance_detections2D.keys()
        ]
        visible_instance_categories = [item.split("|")[0] for item in visible_instances]

        p = inflect.engine()
        visible_instance_counter = {}
        visible_instance_description = ""
        for instance in visible_instances:
            if instance.split("|")[0] not in visible_instance_counter:
                if visible_instance_categories.count(instance.split("|")[0]) > 1:
                    visible_instance_counter[instance.split("|")[0]] = 1
                    visible_instance_description += (
                        f'the {p.ordinal(1)} {split_string_at_capital(instance.split("|")[0]).lower()} at'
                        f" {calculate_boundingbox_center(instance_detections2D[instance])}, "
                    )
                else:
                    visible_instance_description += (
                        f"{get_indefinite_article(split_string_at_capital(instance.split('|')[0])).lower()} at"
                        f" {calculate_boundingbox_center(instance_detections2D[instance])}, "
                    )
            else:
                visible_instance_counter[instance.split("|")[0]] += 1
                visible_instance_description += (
                    f'the {p.ordinal(visible_instance_counter[instance.split("|")[0]])} {split_string_at_capital(instance.split("|")[0]).lower()} at'
                    f" {instance_detections2D[instance]}, "
                )
        visible_instance_description = remove_last_comma(visible_instance_description)

        return replace_last_comma(visible_instance_description)

    def stop(self) -> None:
        raise NotImplementedError


if __name__ == "__main__":
    env = Environment(target_class="LightSwitch")
    print(env.observation_description)
    print(env.act("MoveAhead"))
