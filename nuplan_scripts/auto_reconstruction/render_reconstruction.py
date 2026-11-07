#-------------------------------------------------------------------------------#
# MTGS: Multi-Traversal Gaussian Splatting (https://arxiv.org/abs/2503.12552)   #
# Source code: https://github.com/OpenDriveLab/MTGS                             #
# Copyright (c) OpenDriveLab. All rights reserved.                              #
#-------------------------------------------------------------------------------#
import os
import argparse
from pathlib import Path
os.environ["NERFSTUDIO_DATAPARSER_CONFIGS"] = "nuplan=mtgs.config.nuplan_dataparser:nuplan_dataparser"
os.environ["NERFSTUDIO_METHOD_CONFIGS"] = "mtgs_st=mtgs.config.MTGS_ST:method,mtgs=mtgs.config.MTGS:method"

import nuplan_scripts.utils.ckpt_compat  # noqa: F401  (torch.load weights_only=False for our own checkpoints)
from nuplan_scripts.utils.config import load_config
from nuplan_scripts.utils.constants import CONSOLE

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)

    recon_subdir = os.environ.get('MTGS_RECON_SUBDIR', 'background_reconstruction')
    reconstruction_dir = os.path.join(config.data_root, recon_subdir, config.road_block_name)
    nerfstudio_config = Path(reconstruction_dir) / 'config.yml'

    CONSOLE.log("Rendering background reconstruction model.", style='bold magenta')
    from mtgs.tools.render import RenderInterpolated
    render = RenderInterpolated(
        load_config=nerfstudio_config,
        output_path=Path(reconstruction_dir) / os.environ.get('MTGS_RENDER_DIR', 'render')
    )
    render.main()

    if os.environ.get('MTGS_SKIP_EVAL') != '1':
        CONSOLE.log("Evaluating background reconstruction model.", style='bold magenta')
        import json
        from nerfstudio.utils.eval_utils import eval_setup

        def _skip_init_points(cfg):   # the checkpoint replaces the initial gaussians: no point-cloud filtering
            cfg.pipeline.datamanager.dataparser.load_3D_points = False
            return cfg
        # nerfstudio's ComputePSNR.main with that one config callback
        ns_config, pipeline, checkpoint_path, _ = eval_setup(nerfstudio_config, update_config_callback=_skip_init_points)
        metrics_dict = pipeline.get_average_eval_image_metrics(output_path=None, get_std=True)
        output_path = Path(reconstruction_dir) / 'eval_result.json'
        output_path.write_text(json.dumps({
            "experiment_name": ns_config.experiment_name,
            "method_name": ns_config.method_name,
            "checkpoint": str(checkpoint_path),
            "results": metrics_dict,
        }, indent=2), "utf8")
        CONSOLE.print(f"Saved results to: {output_path}")
