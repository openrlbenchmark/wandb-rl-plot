import wandb
import wandb.apis.reports as wb  # noqa

if __name__ == "__main__":
    env_id = "HalfCheetah-v4"

    wandb.require("report-editing:v0")
    api = wandb.Api()
    report = api.create_report(project="cleanrl")
    report.title = "MuJoCo: rl-games PPO"

    panel_grid = wb.PanelGrid(report)

    run_set2 = wb.RunSet(panel_grid)
    run_set2.name = "rl_games"
    run_set2.entity = "openrlbenchmark"
    run_set2.project = "rl_games"
    run_set2.set_filters_with_python_expr(
        f'params.config.env_config.env_name == "{env_id}" and params.algo.name == "a2c_continuous"'
    )
    run_set2.groupby = ["params.algo.name"]
    panel_grid.run_sets = [run_set2]

    p = wb.LinePlot(panel_grid)
    p.title = env_id
    p.title_x = "Steps"
    p.title_y = "Episodic Return"
    p.max_runs_to_show = 100
    p.x = "global_step"
    p.y = ["rewards/step"]

    panel_grid.panels = [p]

    section2 = [panel_grid]
    report.blocks = section2
    report.save()
