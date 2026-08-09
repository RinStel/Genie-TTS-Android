import logging
import traceback
import os
import contextlib
import importlib.resources
import json

from ...Utils.Constants import PACKAGE_NAME
from ..v2.VITSConverter import VITSConverter
from ..v2.T2SConverter import T2SModelConverter
from ..v2.EncoderConverter import EncoderConverter
from ..v2.Converter import (ENCODER_RESOURCE_PATH, STAGE_DECODER_RESOURCE_PATH,
                            FIRST_STAGE_DECODER_RESOURCE_PATH, T2S_KEYS_RESOURCE_PATH, CACHE_DIR, remove_folder)
from .PromptEncoderConverter import PromptEncoderConverter
from .MultiReferenceExporter import export_prompt_encoder, write_fp16_external_data_manifest

logger = logging.getLogger()

# 使用 V2 ProPlus 的文件。
VITS_RESOURCE_PATH = "Data/v2ProPlus/Models/vits_fp32.onnx"
PROMPT_ENCODER_RESOURCE_PATH = "Data/v2ProPlus/Models/prompt_encoder_fp32.onnx"
VITS_KEYS_RESOURCE_PATH = "./Data/v2ProPlus/Keys/vits_weights.txt"
PROMPT_ENCODER_KEYS_RESOURCE_PATH = "./Data/v2ProPlus/Keys/prompt_encoder_weights.txt"


def convert(
        torch_ckpt_path: str,
        torch_pth_path: str,
        output_dir: str,
        export_multi_reference: bool = False,
        max_reference_count: int = 8,
) -> None:
    # 确保缓存和输出目录存在
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    if len(os.listdir(output_dir)) > 0:
        logger.warning(f"The output directory {output_dir} is not empty!")

    with contextlib.ExitStack() as stack:
        files = importlib.resources.files(PACKAGE_NAME)

        def enter(p: str) -> str:
            return str(stack.enter_context(importlib.resources.as_file(files.joinpath(p))))

        encoder_onnx_path = enter(ENCODER_RESOURCE_PATH)
        stage_decoder_path = enter(STAGE_DECODER_RESOURCE_PATH)
        first_stage_decoder_path = enter(FIRST_STAGE_DECODER_RESOURCE_PATH)
        vits_onnx_path = enter(VITS_RESOURCE_PATH)
        t2s_keys_path = enter(T2S_KEYS_RESOURCE_PATH)
        vits_keys_path = enter(VITS_KEYS_RESOURCE_PATH)
        prompt_encoder_path = enter(PROMPT_ENCODER_RESOURCE_PATH)
        prompt_encoder_keys_path = enter(PROMPT_ENCODER_KEYS_RESOURCE_PATH)

        converter_1 = T2SModelConverter(
            torch_ckpt_path=torch_ckpt_path,
            stage_decoder_onnx_path=stage_decoder_path,
            first_stage_decoder_onnx_path=first_stage_decoder_path,
            key_list_file=t2s_keys_path,
            output_dir=output_dir,
            cache_dir=CACHE_DIR,
        )
        converter_2 = VITSConverter(
            torch_pth_path=torch_pth_path,
            vits_onnx_path=vits_onnx_path,
            key_list_file=vits_keys_path,
            output_dir=output_dir,
            cache_dir=CACHE_DIR,
        )
        converter_3 = EncoderConverter(
            ckpt_path=torch_ckpt_path,
            pth_path=torch_pth_path,
            onnx_input_path=encoder_onnx_path,
            output_dir=output_dir,
        )
        converter_4 = PromptEncoderConverter(
            torch_pth_path=torch_pth_path,
            prompt_encoder_onnx_path=prompt_encoder_path,
            key_list_file=prompt_encoder_keys_path,
            output_dir=output_dir,
            cache_dir=CACHE_DIR,
        )

        try:
            converter_1.run_full_process()
            converter_2.run_full_process()
            converter_3.run_full_process()
            # Keep the distributable model in the same FP16-weight format as
            # the original runtime.  Multi-reference export only adds a graph;
            # it must not create an 88 MB FP32 companion file.
            converter_4.run_full_process(reconstruct_fp32_bin=False)
            write_fp16_external_data_manifest(
                os.path.join(output_dir, "prompt_encoder_fp32.onnx"),
                os.path.join(output_dir, "prompt_encoder_fp16.bin"),
                os.path.join(output_dir, "prompt_encoder_fp16_manifest.json"),
            )
            if export_multi_reference:
                multi_reference_path = os.path.join(output_dir, "prompt_encoder_multi_fp32.onnx")
                spec = export_prompt_encoder(
                    os.path.join(output_dir, "prompt_encoder_fp32.onnx"),
                    multi_reference_path,
                    max_reference_count=max_reference_count,
                    fp16_weight_path=os.path.join(output_dir, "prompt_encoder_fp16.bin"),
                )
                with open(
                        os.path.join(output_dir, "multi_reference_manifest.json"),
                        "w",
                        encoding="utf-8",
                ) as manifest_file:
                    json.dump(
                        {
                            "interface": spec.interface,
                            "max_reference_count": spec.max_reference_count,
                            "max_auxiliary_reference_count": spec.max_auxiliary_reference_count,
                            "input_names": list(spec.input_names),
                            "output_names": list(spec.output_names),
                            "reduction": spec.reduction,
                            "upstream_repository": spec.upstream_repository,
                            "upstream_revision": spec.upstream_revision,
                        },
                        manifest_file,
                        indent=2,
                        sort_keys=True,
                    )
            logger.info(f"🎉 Conversion successful! Saved to: {os.path.abspath(output_dir)}\n"
                        f"- Model Type: V2ProPlus")
        except Exception:
            logger.error(f"❌ A critical error occurred during the conversion process")
            logger.error(traceback.format_exc())
            remove_folder(output_dir)  # 只在失败时清理输出目录
            raise
        finally:
            # 无论成功还是失败，都尝试清理缓存目录
            remove_folder(CACHE_DIR)
