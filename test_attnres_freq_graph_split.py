"""
浅层 AttnRes + 弱频域指导 + 深层 Graph 版本测试入口（无原型提示）。

复用 `test_adapter.py` 的完整测试流程，将编码器替换为
`ImageEncoderViT3dAttnResFreqGraphSplit`。本入口不固定随机种子，
提示点采样行为与 `test.py` 保持一致。

用法示例：
  CUDA_VISIBLE_DEVICES=1 python test_attnres_freq_graph_split.py \
    --data pancreas \
    --snapshot_path "path/to/snapshot_attnres_freq_graph_split/" \
    --attn_res_end_idx 6 \
    --graph_start_idx 6 \
    --graph_variant gravity \
    --graph_aggregation max \
    --graph_gravity_eps 1e-4 \
    --graph_guide_scale 0.1 \
    --freq_guide_scale 0.05 \
    --freq_guidance_source attn_residual \
    --freq_direct_residual_scale 0.0 \
    --data_prefix "/path/to/data/" \
    --checkpoint best
"""

import argparse
import inspect
import sys

import test_adapter as base_test

from modeling.image_encoder_attnres_freq_graph_split import ImageEncoderViT3dAttnResFreqGraphSplit


def _parse_attn_freq_graph_cli_overrides():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--attn_res_end_idx",
        default=6,
        type=int,
        help="前多少层使用 attention residual，需与训练时保持一致",
    )
    parser.add_argument(
        "--graph_guide_scale",
        default=0.1,
        type=float,
        help="浅层 attention residual 对 graph gate 的残差调制幅度，需与训练时保持一致",
    )
    parser.add_argument(
        "--use_freq_guidance",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否启用弱频域指导信号，需与训练时保持一致",
    )
    parser.add_argument(
        "--frequency_bottleneck_dim",
        default=96,
        type=int,
        help="FrequencyAdapter3D bottleneck 维度，需与训练时保持一致",
    )
    parser.add_argument(
        "--freq_guide_scale",
        default=0.05,
        type=float,
        help="频域指导对 graph gate 的残差调制幅度，需与训练时保持一致",
    )
    parser.add_argument(
        "--freq_guidance_source",
        default="attn_residual",
        type=str,
        choices=["attn_residual", "stage_input"],
        help="频域指导输入源，需与训练时保持一致",
    )
    parser.add_argument(
        "--freq_direct_residual_scale",
        default=0.0,
        type=float,
        help="频域增量直接回注主干的强度，需与训练时保持一致",
    )
    args, remaining = parser.parse_known_args(sys.argv[1:])
    sys.argv = [sys.argv[0], *remaining]
    return args


if __name__ == "__main__":
    freq_graph_args = _parse_attn_freq_graph_cli_overrides()

    class AttnResFreqGraphSplitImageEncoderWithCli(ImageEncoderViT3dAttnResFreqGraphSplit):
        """将 split/frequency guidance 参数作为默认值注入，其余图参数由 test_adapter.py 传入。"""

        def __init__(self, *args, **kwargs):
            kwargs.setdefault("attn_res_end_idx", freq_graph_args.attn_res_end_idx)
            kwargs.setdefault("graph_guide_scale", freq_graph_args.graph_guide_scale)
            kwargs.setdefault("use_freq_guidance", freq_graph_args.use_freq_guidance)
            kwargs.setdefault("frequency_bottleneck_dim", freq_graph_args.frequency_bottleneck_dim)
            kwargs.setdefault("freq_guide_scale", freq_graph_args.freq_guide_scale)
            kwargs.setdefault("freq_guidance_source", freq_graph_args.freq_guidance_source)
            kwargs.setdefault("freq_direct_residual_scale", freq_graph_args.freq_direct_residual_scale)
            super().__init__(*args, **kwargs)

    AttnResFreqGraphSplitImageEncoderWithCli.__signature__ = inspect.signature(
        ImageEncoderViT3dAttnResFreqGraphSplit
    )
    base_test.ImageEncoderViT_3d = AttnResFreqGraphSplitImageEncoderWithCli
    base_test.main()
