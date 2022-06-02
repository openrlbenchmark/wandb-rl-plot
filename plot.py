import argparse

import wandb
import wandb.apis.reports as wb  # noqa


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-ids", nargs="+", default=["CartPole-v1", "Acrobot-v1", "MountainCar-v0"],
        help="the ids of the environment to benchmark")
    parser.add_argument("--env-id-key", type=str, default="env_id",
        help="the key of the environment id in the wandb run")
    parser.add_argument("--algo-id-key", type=str, default="exp_name",
        help="the key of the algo id in the wandb run")
    args = parser.parse_args()
    # fmt: on
    return args


if __name__ == "__main__":
    args = parse_args()
    env_id = "BreakoutNoFrameskip-v4"

    wandb.require("report-editing:v0")
    api = wandb.Api()
    report = api.create_report(project="cleanrl")
    report.title = "Atari: CleanRL's PPO"
    report.description = "A comparison of the performance of CleanRL's PPO on Atari games."

    panel_grid = wb.PanelGrid(report)
    run_set1 = wb.RunSet(panel_grid)
    run_set1.name = "CleanRL's ppo_atari.py"
    run_set1.entity = "openrlbenchmark"
    run_set1.project = "cleanrl"
    run_set1.set_filters_with_python_expr(f'env_id == "{env_id}" and exp_name == "ppo_atari"')
    run_set1.groupby = ["exp_name"]
    panel_grid.run_sets = [run_set1]

    p = wb.LinePlot(panel_grid)
    p.title = env_id
    p.title_x = "Steps"
    p.title_y = "Episodic Return"
    p.max_runs_to_show = 100
    p.x = "global_step"
    p.y = ["charts/episodic_return"]

    m = wb.MediaBrowser(panel_grid)
    m.media_keys = "videos"
    m.num_columns = 3

    panel_grid.panels = [p, m]

    section2 = [panel_grid]
    report.blocks = section2
    report.save()
