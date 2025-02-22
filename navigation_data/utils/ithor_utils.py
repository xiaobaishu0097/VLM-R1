import math


def get_ithor_scenes(type: str = "train") -> list:
    if type == "train":
        scenes = (
            [f"FloorPlan{i}" for i in range(1, 21)]
            + [f"FloorPlan{i}" for i in range(201, 221)]
            + [f"FloorPlan{i}" for i in range(301, 321)]
            + [f"FloorPlan{i}" for i in range(401, 421)]
        )
    elif type == "val":
        scenes = (
            [f"FloorPlan{i}" for i in range(21, 26)]
            + [f"FloorPlan{i}" for i in range(221, 226)]
            + [f"FloorPlan{i}" for i in range(321, 326)]
            + [f"FloorPlan{i}" for i in range(421, 426)]
        )
    elif type == "test":
        scenes = (
            [f"FloorPlan{i}" for i in range(26, 31)]
            + [f"FloorPlan{i}" for i in range(226, 231)]
            + [f"FloorPlan{i}" for i in range(326, 331)]
            + [f"FloorPlan{i}" for i in range(426, 431)]
        )
    elif type == "all":
        scenes = (
            [f"FloorPlan{i}" for i in range(1, 31)]
            + [f"FloorPlan{i}" for i in range(201, 231)]
            + [f"FloorPlan{i}" for i in range(301, 331)]
            + [f"FloorPlan{i}" for i in range(401, 431)]
        )
    else:
        raise ValueError(f"Invalid type: {type}")

    return scenes


def calculate_instance_distance(target_info: dict, instance_info: dict) -> float:
    return math.sqrt(
        (target_info["x"] - instance_info["x"]) ** 2
        + (target_info["z"] - instance_info["z"]) ** 2
    )


if __name__ == "__main__":
    print(get_ithor_scenes("val"))
    print(calculate_instance_distance({"x": 1, "z": 1}, {"x": 2, "z": 2}))
