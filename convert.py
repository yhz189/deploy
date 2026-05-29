"""Convert the YOLOv8 ONNX model to RKNN for RK3568.

Run this script in the Ubuntu/rknn-toolkit2 conversion environment:

    python convert.py

The board-side preprocessing sends RGB uint8 images in 0..255 range, so the
RKNN model must include the same 1/255 normalization as the ONNX test script.
That is why std_values is [255, 255, 255].
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

DEFAULT_ONNX = 'best2_fixed.onnx'
DEFAULT_RKNN = 'fridge_yolo_opset11_rknn16.rknn'
FALLBACK_ONNX = (
    Path('best2_fixed.onnx'),
    Path('best2.onnx'),
    Path('yolov8_model/weights/best2_fixed.onnx'),
    Path('yolov8_model/weights/best2.onnx'),
)


def sha256_short(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha256(data).hexdigest()[:16]


def try_print_onnx_info(path: Path) -> None:
    try:
        import onnx
    except Exception as exc:  # pragma: no cover - conversion env helper only
        print(f'[WARN] onnx package unavailable, skip graph info: {exc}')
        return

    model = onnx.load(str(path))
    opsets = ', '.join(
        f'{item.domain or "ai.onnx"}={item.version}'
        for item in model.opset_import
    )
    maxpool_dilations = 0
    for node in model.graph.node:
        if node.op_type != 'MaxPool':
            continue
        for attr in node.attribute:
            if attr.name == 'dilations':
                maxpool_dilations += 1
    print(f'[INFO] ONNX opsets: {opsets}')
    print(f'[INFO] MaxPool dilations attrs: {maxpool_dilations}')
    if maxpool_dilations:
        print('[WARN] RKNN toolkit 1.4.0 may fail on MaxPool dilations; use best2_fixed.onnx or export opset=11.')
    print('[INFO] ONNX inputs:')
    for item in model.graph.input:
        dims = []
        for dim in item.type.tensor_type.shape.dim:
            dims.append(dim.dim_value if dim.dim_value else dim.dim_param or '?')
        print(f'  - {item.name}: {dims}')
    print('[INFO] ONNX outputs:')
    for item in model.graph.output:
        dims = []
        for dim in item.type.tensor_type.shape.dim:
            dims.append(dim.dim_value if dim.dim_value else dim.dim_param or '?')
        print(f'  - {item.name}: {dims}')


def strip_maxpool_dilations(path: Path) -> Path:
    """Remove MaxPool dilations attrs unsupported by RKNN toolkit 1.4.0."""
    try:
        import onnx
    except Exception as exc:  # pragma: no cover - conversion env helper only
        print(f'[WARN] onnx package unavailable, cannot auto-fix dilations: {exc}')
        return path

    model = onnx.load(str(path))
    removed = 0
    for node in model.graph.node:
        if node.op_type != 'MaxPool':
            continue
        kept = []
        for attr in node.attribute:
            if attr.name == 'dilations':
                removed += 1
                print(f'[FIX] Remove MaxPool dilations: {list(attr.ints)}')
            else:
                kept.append(attr)
        if len(kept) != len(node.attribute):
            del node.attribute[:]
            node.attribute.extend(kept)

    if removed == 0:
        return path

    fixed_path = path.with_name(f'{path.stem}_rknn_fixed{path.suffix}')
    onnx.save(model, str(fixed_path))
    print(f'[OK] Saved RKNN-compatible ONNX: {fixed_path} '
          f'(removed {removed} MaxPool dilations attrs)')
    return fixed_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Convert YOLOv8 ONNX to RKNN for RK3568.')
    parser.add_argument('--onnx', default=DEFAULT_ONNX,
                        help='ONNX model path. Prefer best2_fixed.onnx for opset=12 RKNN 1.4.0 conversion, or use opset=11 ONNX.')
    parser.add_argument('--rknn', default=DEFAULT_RKNN,
                        help='Output RKNN model path.')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable RKNN toolkit verbose logs.')
    parser.add_argument('--no-auto-fix', action='store_true',
                        help='Do not remove MaxPool dilations before RKNN load_onnx.')
    return parser.parse_args()


def resolve_onnx_path(text: str) -> Path:
    path = Path(text)
    if path.exists() or text != DEFAULT_ONNX:
        return path
    for candidate in FALLBACK_ONNX:
        if candidate.exists():
            return candidate
    return path


def main() -> None:
    args = parse_args()
    from rknn.api import RKNN

    onnx_model = resolve_onnx_path(args.onnx)
    rknn_model = Path(args.rknn)

    if not onnx_model.exists():
        raise FileNotFoundError(f'ONNX model not found: {onnx_model}')

    print(f'[INFO] ONNX: {onnx_model.resolve()}')
    print(f'[INFO] ONNX sha256: {sha256_short(onnx_model)}')
    try_print_onnx_info(onnx_model)
    if not args.no_auto_fix:
        onnx_model = strip_maxpool_dilations(onnx_model)
        if onnx_model.exists():
            print(f'[INFO] RKNN input ONNX: {onnx_model.resolve()}')

    rknn = RKNN(verbose=args.verbose)
    try:
        ret = rknn.config(
            mean_values=[[0, 0, 0]],
            std_values=[[255, 255, 255]],
            target_platform='rk3568',
        )
        assert ret == 0, f'config failed: {ret}'
        print('[OK] RKNN config: RGB uint8 input, divide by 255, target rk3568')

        ret = rknn.load_onnx(model=str(onnx_model))
        assert ret == 0, f'load_onnx failed: {ret}'
        print('[OK] ONNX loaded')

        ret = rknn.build(do_quantization=False)
        assert ret == 0, f'build failed: {ret}'
        print('[OK] RKNN built without int8 quantization')

        ret = rknn.export_rknn(str(rknn_model))
        assert ret == 0, f'export_rknn failed: {ret}'
        print(f'[OK] RKNN exported: {rknn_model.resolve()}')
    finally:
        rknn.release()


if __name__ == '__main__':
    main()
