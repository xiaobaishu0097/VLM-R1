import json
import os
import random
import sys
from typing import Any, Optional

import h5py
import networkx as nx
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from environment import Environment
from scene import iTHORPosition
from utils.detection_utils import calculate_boundingbox_size
from utils.visualization_utils import identity


class OfflineEnvironment(Environment):
    def __init__(
        self,
        configs: dict,
        scene: Optional[str] = None,
        target_class: str = "ArmChair",
        # Sensor parameters
        always_include_observed_instances: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            configs=configs,
            scene=scene,
            target_class=target_class,
            always_include_observed_instances=always_include_observed_instances,
            **kwargs,
        )

        self.environment_data_dir = configs["environment"]["offline_environment"][
            "environment_data_dir"
        ]

        if scene is not None:
            self.random_reset(scene=scene)

    def load_scene_data(self, scene: str) -> None:
        if hasattr(self, "_controller"):
            self._controller.close()
        self._controller = h5py.File(
            os.path.join(self.environment_data_dir, f"{scene}.h5"), "r"
        )
        self._scene_graph = nx.node_link_graph(
            json.loads(self._controller["scene_info"]["scene_graph"][()]), edges="links"
        )
        self._object_visibility_map = json.loads(
            self._controller["scene_info"]["object_visibility_map"][()]
        )

    def close(self):
        self._controller.close()

    def random_reset(self, scene: str):
        self._scene = scene
        self.load_scene_data(scene=scene)

        positions = list(self._scene_graph.nodes)
        self._position = random.choice(positions)

        if hasattr(self, "target_direction"):
            del self.target_direction
            del self.last_succeeded_action
        if hasattr(self, "target_is_observed"):
            del self.target_is_observed
        if hasattr(self, "_closest_target_id"):
            del self._closest_target_id
        if hasattr(self, "_potential_target_ids"):
            del self._potential_target_ids

    @property
    def scene(self) -> str:
        if not hasattr(self, "_scene"):
            raise ValueError("Scene is not set.")
        return self._scene

    @property
    def scene_id(self) -> str:
        return self._scene

    def teleport_agent(self, position: iTHORPosition) -> None:
        self._position = str(position)

    def get_object_visibility_map(
        self,
        progress_function: Any = identity,
        progress_description: Optional[str] = None,
    ) -> dict:
        return self._object_visibility_map

    @property
    def observation(self) -> np.ndarray:
        return self._controller["scene_states"][str(self.position)]["observation"][()]

    @property
    def position(self) -> iTHORPosition:
        if not hasattr(self, "_position"):
            raise ValueError("Position is not set.")
        return iTHORPosition(self._position)

    @property
    def step_metadata(self) -> dict:
        return self._controller["scene_states"][str(self.position)]["metadata"]

    @property
    def step_agent_metadata(self) -> h5py.File:
        return self._controller["scene_states"][str(self.position)]["metadata"]["agent"]

    @property
    def step_agent_position_metadata(self) -> dict:
        return {
            "x": json.loads(
                self._controller["scene_states"][str(self.position)]["metadata"][
                    "agent"
                ]["position"]["x"][()]
            ),
            "y": json.loads(
                self._controller["scene_states"][str(self.position)]["metadata"][
                    "agent"
                ]["position"]["y"][()]
            ),
            "z": json.loads(
                self._controller["scene_states"][str(self.position)]["metadata"][
                    "agent"
                ]["position"]["z"][()]
            ),
        }

    @property
    def step_agent_rotation_metadata(self) -> dict:
        return {
            "x": json.loads(
                self._controller["scene_states"][str(self.position)]["metadata"][
                    "agent"
                ]["rotation"]["x"][()]
            ),
            "y": json.loads(
                self._controller["scene_states"][str(self.position)]["metadata"][
                    "agent"
                ]["rotation"]["y"][()]
            ),
            "z": json.loads(
                self._controller["scene_states"][str(self.position)]["metadata"][
                    "agent"
                ]["rotation"]["z"][()]
            ),
        }

    @property
    def step_agent_cameraHorizon_metadata(self) -> int:
        return json.loads(
            self._controller["scene_states"][str(self.position)]["metadata"]["agent"][
                "cameraHorizon"
            ][()]
        )

    @property
    def step_object_metadata(self) -> list[dict]:
        return json.loads(
            self._controller["scene_states"][str(self.position)]["metadata"]["objects"][
                ()
            ]
        )

    @property
    def observation_detection(self) -> dict:
        return json.loads(
            self._controller["scene_states"][str(self.position)][
                "observation_detection"
            ][()]
        )

    def act(self, action: str) -> bool:
        if action == "MoveAhead":
            rotation = self.position.rotation_y
            if rotation == 0:
                moved_x = self.position.position_x
                moved_z = self.position.position_z + 0.25
            elif rotation == 90:
                moved_x = self.position.position_x + 0.25
                moved_z = self.position.position_z
            elif rotation == 180:
                moved_x = self.position.position_x
                moved_z = self.position.position_z - 0.25
            elif rotation == 270:
                moved_x = self.position.position_x - 0.25
                moved_z = self.position.position_z
            elif rotation == 45:
                moved_x = self.position.position_x + 0.25
                moved_z = self.position.position_z + 0.25
            elif rotation == 135:
                moved_x = self.position.position_x + 0.25
                moved_z = self.position.position_z - 0.25
            elif rotation == 225:
                moved_x = self.position.position_x - 0.25
                moved_z = self.position.position_z - 0.25
            elif rotation == 315:
                moved_x = self.position.position_x - 0.25
                moved_z = self.position.position_z + 0.25

            moved_position = f"{moved_x:0.2f}|{moved_z:0.2f}|{self.position.rotation_y:d}|{self.position.horizon:d}"
            if moved_position not in self.scene_graph.nodes:
                return False

        elif action == "RotateLeft":
            moved_position = "{:0.2f}|{:0.2f}|{:d}|{:d}".format(
                self.position.position_x,
                self.position.position_z,
                round((self.position.rotation_y - 45) % 360),
                round(self.position.horizon),
            )

        elif action == "RotateRight":
            moved_position = "{:0.2f}|{:0.2f}|{:d}|{:d}".format(
                self.position.position_x,
                self.position.position_z,
                round((self.position.rotation_y + 45) % 360),
                round(self.position.horizon),
            )

        elif action == "LookUp":
            if (
                self.step_agent_cameraHorizon_metadata == 0
                or self.step_agent_cameraHorizon_metadata == 30
            ):
                moved_position = "{:0.2f}|{:0.2f}|{:d}|{:d}".format(
                    self.position.position_x,
                    self.position.position_z,
                    round(self.position.rotation_y),
                    round(self.position.horizon - 30),
                )
            else:
                return False

        elif action == "LookDown":
            if (
                self.step_agent_cameraHorizon_metadata == 0
                or self.step_agent_cameraHorizon_metadata == -30
            ):
                moved_position = "{:0.2f}|{:0.2f}|{:d}|{:d}".format(
                    self.position.position_x,
                    self.position.position_z,
                    round(self.position.rotation_y),
                    round(self.position.horizon + 30),
                )
            else:
                return False

        elif action == "Done":
            return self.check_done_status()

        if moved_position in self.scene_graph.neighbors(self._position):
            self.last_succeeded_action = action
            self._position = moved_position
            return True
        return False

    def get_positions(
        self,
        n_max_positions: Optional[int] = None,
        detailed_position: bool = False,
        **kwargs,
    ) -> list[iTHORPosition]:
        positions = list(self._scene_graph.nodes)

        if n_max_positions is not None and n_max_positions < len(positions):
            positions = random.sample(positions, k=n_max_positions)
        positions = [iTHORPosition(position) for position in positions]

        return positions

    def is_valid_episode(self, navigation_episode: list, target_class: str) -> bool:
        final_state = iTHORPosition(navigation_episode[-1])
        self.teleport_agent(final_state)

        instance_detection = self.observation_detection

        for instance_id in instance_detection:
            if (
                instance_id.split("|")[0] == target_class
                and calculate_boundingbox_size(instance_detection[instance_id])
                > self.boundingbox_size_threshold
            ):
                return True
            elif (
                instance_id.split("|")[0] == target_class
                and calculate_boundingbox_size(instance_detection[instance_id])
                <= self.boundingbox_size_threshold
                and str(final_state) in self.object_visibility_map[instance_id]
            ):
                self.object_visibility_map[instance_id].remove(str(final_state))

        return False


if __name__ == "__main__":
    from utils.config import load_config

    configs = load_config("./configs/__base__/base_environment.yaml")
    configs["experiment_id"] = "debugging"
    offline_environment = OfflineEnvironment(configs=configs, scene="FloorPlan212")

    print(offline_environment.scene)

    offline_environment.random_reset(scene="FloorPlan212")
    print(offline_environment.position)
    offline_environment.random_reset(scene="FloorPlan212")
    print(offline_environment.position)

    offline_environment.teleport_agent(
        iTHORPosition(str(random.choice(offline_environment.get_positions())))
    )
    print(offline_environment.position)

    print(offline_environment.observation)
    print(offline_environment.observation_detection)
    print(offline_environment.step_object_metadata)
    print(offline_environment.step_agent_cameraHorizon_metadata)
