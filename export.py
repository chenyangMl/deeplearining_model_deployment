import torch
import numpy as np
import os, sys
import pkg_resources as pkg
from pathlib import Path
import argparse

sys.path.insert(0, os.getcwd())

from mainareaDet.models.yolo import  Detect

def check_version(current='0.0.0', minimum='0.0.0', name='version ', pinned=False, hard=False, verbose=False):
    # Check version vs. required version
    current, minimum = (pkg.parse_version(x) for x in (current, minimum))
    result = (current == minimum) if pinned else (current >= minimum)  # bool
    s = f'WARNING ⚠️ {name}{minimum} is required by YOLOv5, but {name}{current} is currently installed'  # string
    return result

def colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
    colors = {
        'black': '\033[30m',  # basic colors
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'magenta': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m',
        'bright_black': '\033[90m',  # bright colors
        'bright_red': '\033[91m',
        'bright_green': '\033[92m',
        'bright_yellow': '\033[93m',
        'bright_blue': '\033[94m',
        'bright_magenta': '\033[95m',
        'bright_cyan': '\033[96m',
        'bright_white': '\033[97m',
        'end': '\033[0m',  # misc
        'bold': '\033[1m',
        'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']

def export_onnx(model, im, file, opset, dynamic, simplify, prefix=colorstr('ONNX:')):
    # YOLOv5 ONNX export
    import onnx

    print(f'\n{prefix} starting export with onnx {onnx.__version__}...')
    f = file.with_suffix('.onnx')

    output_names = ['output0']
    if dynamic:
        dynamic = {'images': {0: 'batch', 2: 'height', 3: 'width'}}  # shape(1,3,640,640)
        dynamic['output0'] = {0: 'batch', 1: 'anchors'}  # shape(1,25200,85)

    torch.onnx.export(
        model.cpu() if dynamic else model,  # --dynamic only compatible with cpu
        im.cpu() if dynamic else im,
        f,
        verbose=False,
        opset_version=opset,
        do_constant_folding=True,
        input_names=['images'],
        output_names=output_names,
        dynamic_axes=dynamic or None)

    # Checks
    model_onnx = onnx.load(f)  # load onnx model
    onnx.checker.check_model(model_onnx)  # check onnx model

    # Metadata
    d = {'stride': int(max(model.stride)), 'names': model.names}
    for k, v in d.items():
        meta = model_onnx.metadata_props.add()
        meta.key, meta.value = k, str(v)
    onnx.save(model_onnx, f)

    # Simplify
    if simplify:
        try:
            cuda = torch.cuda.is_available()
            # check_requirements(('onnxruntime-gpu' if cuda else 'onnxruntime', 'onnx-simplifier>=0.4.1'))
            import onnxsim

            print(f'{prefix} simplifying with onnx-simplifier {onnxsim.__version__}...')
            model_onnx, check = onnxsim.simplify(model_onnx)
            assert check, 'assert check failed'
            onnx.save(model_onnx, f)
        except Exception as e:
            print(f'{prefix} simplifier failure: {e}')
    print(f'\n{prefix} save onnx to {f}.')
    return f, model_onnx


def export_engine(model, im, file, half, dynamic, simplify, workspace=4, verbose=False, prefix=colorstr('TensorRT:')):
    # YOLOv5 TensorRT export https://developer.nvidia.com/tensorrt
    assert im.device.type != 'cpu', 'export running on CPU but must be on GPU, i.e. `python export.py --device 0`'
    try:
        import tensorrt as trt
    except Exception:
        print("Error: import tensorrt")

    if trt.__version__[0] == '7':  # TensorRT 7 handling https://github.com/ultralytics/yolov5/issues/6012
        grid = model.det.anchor_grid
        model.det.anchor_grid.anchor_grid = [a[..., :1, :1, :] for a in grid]
        export_onnx(model, im, file, 12, dynamic, simplify)  # opset 12
        model.det.anchor_grid = grid
    else:  # TensorRT >= 8
        check_version(trt.__version__, '8.0.0', hard=True)  # require tensorrt>=8.0.0
        export_onnx(model, im, file, 12, dynamic, simplify)  # opset 12
    onnx = file.with_suffix('.onnx')

    print(f'\n{prefix} starting export with TensorRT {trt.__version__}...')
    assert onnx.exists(), f'failed to export ONNX file: {onnx}'
    f = file.with_suffix('.engine')  # TensorRT engine file
    logger = trt.Logger(trt.Logger.INFO)
    if verbose:
        logger.min_severity = trt.Logger.Severity.VERBOSE

    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    config.max_workspace_size = workspace * 1 << 30
    # config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace << 30)  # fix TRT 8.4 deprecation notice

    flag = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx)):
        raise RuntimeError(f'failed to load ONNX file: {onnx}')

    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    outputs = [network.get_output(i) for i in range(network.num_outputs)]
    for inp in inputs:
        print(f'{prefix} input "{inp.name}" with shape{inp.shape} {inp.dtype}')
    for out in outputs:
        print(f'{prefix} output "{out.name}" with shape{out.shape} {out.dtype}')

    if dynamic: #这里的动态输入，其实只是batch纬度动态，并不是H，W纬度动态
        if im.shape[0] <= 1:
            print(f"{prefix} WARNING ⚠️ --dynamic model requires maximum --batch-size argument")
        profile = builder.create_optimization_profile()
        for inp in inputs: #这里只是设置了batchsize的最小到最大范围。(1,3,224,224), (4,3,224,224), (8,3,224, 224)
            profile.set_shape(inp.name, (1, *im.shape[1:]), (max(1, im.shape[0] // 2), *im.shape[1:]), im.shape)
        config.add_optimization_profile(profile)

    print(f'{prefix} building FP{16 if builder.platform_has_fast_fp16 and half else 32} engine as {f}')
    if builder.platform_has_fast_fp16 and half:
        config.set_flag(trt.BuilderFlag.FP16)
    with builder.build_engine(network, config) as engine, open(f, 'wb') as t:
        t.write(engine.serialize())
    print(f"{prefix} Infor save model to {f}")
    return f, None

def load_model(modelPath, netCfg="mainareaDet/models/polygon_yolov5n.yaml", nc=2, fp16=0):
    from mainareaDet.models.yolo import Polygon_Model
    model = Polygon_Model(cfg=netCfg, ch=3, nc=nc)
    statedict0 = torch.load(modelPath)
    model.load_state_dict(statedict0)
    model.eval()
    if fp16:
        model.half()
    return model

def main(args):
    modelPath = Path(args.modelPath)
    assert modelPath.exists(), "Not found modelPath = %s"%modelPath
    batchsize = args.bs
    inshape   = (args.imgsz, args.imgsz)
    
    device = torch.device("cpu") if args.device <0 else torch.device("cuda")
    model = load_model(modelPath)
    model.eval().to(device)
    im = torch.zeros(batchsize, 3, *inshape).to(device)  # image size(1,3,320,192) BCHW iDetection

    if args.export.endswith("onnx"):
        file = modelPath.with_suffix(".onnx")
        export_onnx(model, im, file, opset=12, dynamic=args.dynamic, simplify=False)
    elif args.export.endswith("engine"):
        file = modelPath.with_suffix(".engine")
        export_engine(model, im, file, half=False, dynamic=args.dynamic, simplify=False)

def argConfig():
    parser = argparse.ArgumentParser(description='main.py')
    parser.add_argument('--modelPath', type=str, default="weights/best.pth", help="加载的模型 (pth)")
    parser.add_argument('--export', type=str, default="onnx",choices=["onnx", "engine"], help="输出的模型")
    parser.add_argument('--imgsz', type=int, default=224, help="输入图片大小")
    parser.add_argument('--bs','--batch-size', type=int, default=1, help="batch size")
    parser.add_argument('--dynamic', action='store_true', help='ONNX/TF/TensorRT: dynamic axes')
    parser.add_argument('--device', type=int, default=-1, help='cuda(cuda:0) or cpu(-1)')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    """Usage
        转化成onnx模型: python export.py --export onnx
        转化成engine模型(tensorrt): python export.py --export engine --device 0
        转化成engine模型(tensorrt, 动态batch): python export.py --export engine --bs 32 --device 0 --dynamic
    """
    args = argConfig()
    main(args)