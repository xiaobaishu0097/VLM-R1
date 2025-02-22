import argparse
import copy
import json
import os
import random
import sys
import time
import traceback
from typing import Optional

import h5py
import networkx as nx
import torch.multiprocessing as mp
from PIL import Image
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    track,
)
from torch.multiprocessing import Event, Manager

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from offline_environment import OfflineEnvironment as Environment
from scene import iTHORPosition, iTHORScene
from utils.config import load_config
from utils.debug_utils import is_debug_mode
from utils.ithor_utils import get_ithor_scenes
from utils.logger import Logger


def optimal_navigation_regulation_parser(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--scenes", nargs="+", type=str, default=None)
    parser.add_argument("--scene-type", type=str, default="train")

    parser.add_argument("--always-include-observed-instances", action="store_true")

    parser.add_argument("--n-max-targets", type=int, default=10000)
    parser.add_argument("--n-max-positions", type=int, default=10000)
    parser.add_argument("--sample-detailed-position", action="store_true")

    parser.add_argument("--num-processes", type=int, default=1)
    parser.add_argument("--gpu-ids", nargs="+", type=int, default=None)

    parser.add_argument("--output-dir", type=str, default="./work_dirs/debug")
    parser.add_argument("--log-file", type=str, default="log.txt")

    parser.add_argument("--debug", action="store_true")

    return parser


def generate_optimal_navigation_regulations(
    args: argparse.Namespace,
    configs: dict,
    scene: str,
    gpu_id: Optional[int] = None,
    progress: Optional[dict] = None,
    done_event: Optional[Event] = None,
    debug: bool = False,
    **kwargs,
) -> None:
    logger = Logger.get_logger(
        name="ithor_episode_instructions",
    )

    try:
        if gpu_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_id}"

        configs["environment"]["offline_environment"][
            "environment_data_dir"
        ] = args.data_dir
        environment = Environment(scene=scene, configs=configs)

        scene_id = int(scene.replace("FloorPlan", ""))

        with h5py.File(os.path.join(args.data_dir, f"{scene}.h5"), "r") as rf:
            target_ids = json.loads(rf["scene_info"]["target_ids"][()])
            object_visibility_size = json.loads(
                rf["scene_info"]["object_visibility_size"][()]
            )

        target_classes: list[str] = sorted(
            list(set([item.split("|")[0] for item in target_ids]))
        )

        n_instruction = 0
        navigation_data = []
        positions: list[iTHORPosition] = environment.get_positions(
            detailed_position=args.sample_detailed_position
        )
        for target_class in target_classes:
            start_positions = random.sample(
                positions, min(len(positions), args.n_max_positions)
            )

            visible_positions = environment.get_target_class_visible_positions(
                target_class
            )

            for start_position in start_positions:
                navigation_episodes = environment.estimate_shortest_path(
                    str(start_position), environment.scene_graph, visible_positions
                )

                if (
                    len(navigation_episodes) == 0
                    or navigation_episodes[-1] not in visible_positions
                    or not environment.is_valid_episode(
                        navigation_episodes, target_class
                    )
                ):
                    # TODO: log all these kind of navigation episodes
                    candidate_positions = [
                        item for item in positions if item not in start_positions
                    ]
                    if len(candidate_positions) > 0:
                        start_positions.append(
                            random.choice(candidate_positions),
                        )
                    else:
                        n_instruction += 1
                    continue

                target_id = None
                target_bbox_size = None
                for target in object_visibility_size[navigation_episodes[-1]]:
                    if (
                        target_class == target.split("|")[0]
                        and navigation_episodes[-1]
                        in environment.object_visibility_map[target]
                        and object_visibility_size[navigation_episodes[-1]][target]
                        > 100
                        and (
                            target_id is None
                            or target_bbox_size
                            < object_visibility_size[navigation_episodes[-1]][target]
                        )
                    ):
                        target_id = target
                        target_bbox_size = object_visibility_size[
                            navigation_episodes[-1]
                        ][target]

                assert (
                    target_id is not None
                ), f"Target id is None {navigation_episodes[-1]}"

                environment.set_target_id(target_id)

                historical_actions = []
                for i_steps, position in enumerate(navigation_episodes):
                    instruction_id = f"{scene_id:05d}-{n_instruction:05d}-{i_steps:03d}"

                    target_relative_position = (
                        environment.get_target_relative_position()
                    )

                    agent_observation = environment.observation
                    # convert numpy array to PIL image
                    agent_observation = Image.fromarray(agent_observation)

                    optimal_action = environment.optimal_action
                    navigation_step = {
                        "height": environment.observation_height,
                        "width": environment.observation_width,
                        "id": copy.deepcopy(instruction_id),
                        "dataset_name": "ithor",
                        "image_name": f"{instruction_id}.png",
                        "position": copy.deepcopy(position),
                        "optimal_action": copy.deepcopy(optimal_action),
                        "historical_actions": copy.deepcopy(historical_actions),
                        "target_id": copy.deepcopy(target_id),
                        "target_relative_position": copy.deepcopy(
                            target_relative_position
                        ),
                    }

                    assert not (
                        len(navigation_episodes[i_steps:]) == 1
                        and optimal_action != "Done"
                    ), (
                        f"Error in {i_steps} Step in {navigation_episodes} with"
                        f" {optimal_action}"
                    )

                    if not os.path.exists(
                        os.path.join(args.output_dir, "observations")
                    ):
                        os.makedirs(os.path.join(args.output_dir, "observations"))

                    # save image array using PIL
                    agent_observation.save(
                        os.path.join(
                            args.output_dir,
                            "observations",
                            f"{instruction_id}.png",
                        )
                    )

                    navigation_data.append(navigation_step)
                    historical_actions.append(optimal_action)

                n_instruction += 1
                if progress is not None:
                    progress[scene] = (
                        n_instruction,
                        len(target_classes) * min(len(positions), args.n_max_positions),
                    )

    except Exception as e:
        exc_type, _, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]

        logger.error(
            f"Error is {e} \n"
            + f"Error in scene {scene} \n"
            + f"{exc_type} {fname} {exc_tb.tb_lineno} \n"
            + "".join(traceback.format_exception(*sys.exc_info()))
        )

    finally:
        if not os.path.exists(os.path.join(args.output_dir, "annotations")):
            os.makedirs(os.path.join(args.output_dir, "annotations"))
        with open(
            os.path.join(
                args.output_dir,
                "annotations",
                "navigation_data.json",
            ),
            "w",
        ) as f:
            json.dump(navigation_data, f, indent=4)

        time.sleep(1)
        if done_event is not None:
            done_event.set()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = optimal_navigation_regulation_parser(parser)
    args = parser.parse_args()

    debug = is_debug_mode() or args.debug

    configs = load_config(args.config)
    debug = is_debug_mode() or args.debug

    console = Console()

    configs["output_dir"] = args.output_dir
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    configs["experiment_id"] = "offline_ithor_environment_generation"
    logger = Logger.get_logger(
        name=configs["experiment_id"],
        file_name=os.path.join(args.output_dir, args.log_file),
    )
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    logger = Logger.get_logger(
        name="ithor_episode_instructions",
        file_name=os.path.join(args.output_dir, args.log_file),
    )

    if args.scenes is None:
        scenes = get_ithor_scenes(args.scene_type)
    else:
        scenes = args.scenes

    if not debug:
        progress_columns = (
            SpinnerColumn(),
            "[progress.description]{task.description}",
            BarColumn(),
            TaskProgressColumn(),
            "Elapsed:",
            TimeElapsedColumn(),
            "Remaining:",
            TimeRemainingColumn(),
        )
        mp.set_start_method("spawn", force=True)

        manager = Manager()
        progress_dict = manager.dict()
        done_events = {scene: manager.Event() for scene in scenes}

        with Progress(*progress_columns) as progress:
            task_id = progress.add_task(
                f"[cyan]Generate {len(scenes)} Scenes...", total=len(scenes)
            )

            def callback(*a):
                progress.update(task_id, advance=1, refresh=True)

            def error_callback(e):
                print(f"Error: {e}")

            tasks = {}
            pool = mp.Pool(args.num_processes)

            for i, scene in enumerate(scenes):
                if i < args.num_processes:
                    time.sleep(5)
                else:
                    time.sleep(2)
                pool.apply_async(
                    generate_optimal_navigation_regulations,
                    args=(
                        args,
                        configs,
                        scene,
                        (
                            args.gpu_ids[i % len(args.gpu_ids)]
                            if args.gpu_ids is not None
                            else None
                        ),
                        progress_dict,
                        done_events[scene],
                    ),
                    callback=callback,
                    error_callback=error_callback,
                )

            task_dict = {}
            while not all(event.is_set() for event in done_events.values()):
                for scene, event in done_events.items():
                    if (
                        scene not in task_dict
                        and not event.is_set()
                        and scene in progress_dict
                    ):
                        task_dict[scene] = progress.add_task(
                            f"[red] Generate {scene} {progress_dict[scene][1]} data",
                            total=progress_dict[scene][1],
                        )
                    if scene in task_dict:
                        progress.update(
                            task_dict[scene], completed=progress_dict[scene][0]
                        )
                        if event.is_set():
                            progress.remove_task(task_dict[scene])
                            del task_dict[scene]

                time.sleep(1)
            progress.stop()

            pool.close()
            pool.join()
    else:
        for scene in track(scenes):
            generate_optimal_navigation_regulations(args, configs, scene, debug=debug)

    logger.info("Optimal Navigation Regulations Generation Completed.")
