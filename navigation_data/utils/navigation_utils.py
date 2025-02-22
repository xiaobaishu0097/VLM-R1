import json

from evaluation.metrics import compute_evaluation_metrices
from utils.logger import Logger


def present_navigation_results(results: dict, configs: dict) -> None:
    logger = Logger.get_logger(name=configs["experiment_id"])

    episode_results = {"success": {}, "length": {}, "optimal_length": {}}
    for scene, episode_info in results.items():
        for episode_result in episode_info:
            if scene.split("_")[0] not in episode_results["success"]:
                episode_results["success"][scene.split("_")[0]] = []
                episode_results["length"][scene.split("_")[0]] = []
                episode_results["optimal_length"][scene.split("_")[0]] = []
            episode_results["success"][scene.split("_")[0]].append(episode_result[0])
            episode_results["length"][scene.split("_")[0]].append(episode_result[1])
            episode_results["optimal_length"][scene.split("_")[0]].append(
                episode_result[2]
            )

    spls, srs = compute_evaluation_metrices(
        [
            item
            for scene in episode_results["success"]
            for item in episode_results["success"][scene]
        ],
        [
            item
            for scene in episode_results["length"]
            for item in episode_results["length"][scene]
        ],
        [
            item
            for scene in episode_results["optimal_length"]
            for item in episode_results["optimal_length"][scene]
        ],
    )

    logger.info(
        f"{configs['experiment_id']}: Navigation Success Rate:"
        f" {sum([item for scene in episode_results['success'] for item in episode_results['success'][scene]])} /"
        f" {configs['experiment']['n_experiments']}\nSPL:"
        f" {json.dumps(spls, indent=4)}\nSR: {json.dumps(srs, indent=4)}"
    )
    logger.info("Navigation Completed.")
