import os
from copy import deepcopy
import collections

import torch
from tqdm import tqdm

import val as validate

try:
    from pytorch_quantization import nn as quant_nn
    from pytorch_quantization import calib
    from pytorch_quantization.tensor_quant import QuantDescriptor
    from pytorch_quantization import quant_modules
except ImportError:
    raise ImportError(
        "pytorch-quantization is not installed. Install from "
        "https://github.com/NVIDIA/TensorRT/tree/master/tools/pytorch-quantization."
    )

def collect_stats(model, data_loader, num_batches, device):
    """Feed data to the network and collect statistics"""
    # Enable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
            else:
                module.disable()

    # Feed data to the network for collecting stats
    for i, (img, _, _, _) in tqdm(enumerate(data_loader), total=num_batches):
        img = img.to(device, non_blocking=True).float() / 255.0        
        model(img)
        if i >= num_batches:
            break

    # Disable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.enable_quant()
                module.disable_calib()
            else:
                module.enable()


def compute_amax(model, **kwargs):
    # Load calib result
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                if isinstance(module._calibrator, calib.MaxCalibrator):
                    module.load_calib_amax()
                else:
                    module.load_calib_amax(**kwargs)
            # print(F"{name:40}: {module}")
    model.cuda()


def calibrate_model(model, model_name, data_loader, num_calib_batch, calibrator, hist_percentile, out_dir, device):
    """
        Feed data to the network and calibrate.
        Arguments:
            model: classification model
            model_name: name to use when creating state files
            data_loader: calibration data set
            num_calib_batch: amount of calibration passes to perform
            calibrator: type of calibration to use (max/histogram)
            hist_percentile: percentiles to be used for historgram calibration
            out_dir: dir to save state files in
    """

    if num_calib_batch > 0:
        print("Calibrating model")
        with torch.no_grad():
            collect_stats(model, data_loader, num_calib_batch, device)

        if not calibrator == "histogram":
            compute_amax(model, method="max")
            calib_output = os.path.join(
                out_dir,
                F"{model_name}-max-{num_calib_batch*data_loader.batch_size}.pth")

            ckpt = {'model': deepcopy(model)}
            torch.save(ckpt, calib_output)
        else:
            for percentile in hist_percentile:
                print(F"{percentile} percentile calibration")
                compute_amax(model, method="percentile")
                calib_output = os.path.join(
                    out_dir,
                    F"{model_name}-percentile-{percentile}-{num_calib_batch*data_loader.batch_size}.pth")

                ckpt = {'model': deepcopy(model)}
                torch.save(ckpt, calib_output)

            for method in ["mse", "entropy"]:
                print(F"{method} calibration")
                compute_amax(model, method=method)

                calib_output = os.path.join(
                    out_dir,
                    F"{model_name}-{method}-{num_calib_batch*data_loader.batch_size}.pth")

                ckpt = {'model': deepcopy(model)}
                torch.save(ckpt, calib_output)


def evaluate_accuracy(model, opt, testloader, data_dict):
    results, maps, _ = validate.run(data=data_dict,
                                    half=False,
                                    model=model,
                                    dataloader=testloader,                                    
                                    plots=False)

    map50 = list(results)[2]
    map = list(results)[3]
    return map50, map


def build_sensitivity_profile(model, opt, testloader, data_dict):
    quant_layer_names = []
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            module.disable()
            layer_name = name.replace("._input_quantizer", "").replace("._weight_quantizer", "")
            if layer_name not in quant_layer_names:
                quant_layer_names.append(layer_name)
    print(F"{len(quant_layer_names)} quantized layers found.")

    # Build sensitivity profile
    quant_layer_sensitivity = {}
    for i, quant_layer in enumerate(quant_layer_names):
        print(F"Enable {quant_layer}")
        for name, module in model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer) and quant_layer in name:
                module.enable()
                print(F"{name:40}: {module}")

        # Eval the model
        map50, map50_95 = evaluate_accuracy(model, opt, testloader, data_dict)
        print(F"mAP@IoU=0.50: {map50}, mAP@IoU=0.50:0.95: {map50_95}")
        quant_layer_sensitivity[quant_layer] = opt.accu_tolerance - (map50 + map50_95)

        for name, module in model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer) and quant_layer in name:
                module.disable()
                print(F"{name:40}: {module}")

    # Skip most sensitive layers until accuracy target is met
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            module.enable()
    quant_layer_sensitivity = collections.OrderedDict(sorted(quant_layer_sensitivity.items(), key=lambda x: x[1]))
    print(quant_layer_sensitivity)

    # skipped_layers = []
    # for quant_layer, _ in quant_layer_sensitivity.items():
    #     for name, module in model.named_modules():
    #         if isinstance(module, quant_nn.TensorQuantizer):
    #             if quant_layer in name:
    #                 print(F"Disable {name}")
    #                 if not quant_layer in skipped_layers:
    #                     skipped_layers.append(quant_layer)
    #                 module.disable()
    #     map50, map50_95 = evaluate_accuracy(model, opt, testloader, data_dict)
    #     if (map50 + map50_95) >= opt.accu_tolerance - 0.05:
    #         print(F"Accuracy tolerance {opt.accu_tolerance} is met by skipping {len(skipped_layers)} sensitive layers.")
    #         print(skipped_layers)
    #         onnx_filename = opt.ckpt_path.replace('.pt', F'_skip{len(skipped_layers)}.onnx')
    #         export_onnx(model, onnx_filename, opt.batch_size_onnx, opt.dynamic)
    #         return
    # raise ValueError(f"Accuracy tolerance {opt.accu_tolerance} can not be met with any layer quantized!")


def skip_sensitive_layers(model, opt, testloader):
    print('Skip the sensitive layers.')
    # Sensitivity layers for yolov5s
    skipped_layers = ['model.1.conv',          # the first conv
                      'model.2.cv1.conv',      # the second conv
                      'model.24.m.2',          # detect layer
                      'model.24.m.1',          # detect layer
                      ]

    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            layer_name = name.replace("._input_quantizer", "").replace("._weight_quantizer", "")
            if layer_name in skipped_layers:
                print(F"Disable {name}")
                module.disable()

    map50, map50_95 = evaluate_accuracy(model, opt, testloader)
    print(F"mAP@IoU=0.50: {map50}, mAP@IoU=0.50:0.95: {map50_95}")

    onnx_filename = opt.ckpt_path.replace('.pt', F'_skip{len(skipped_layers)}.onnx')
    export_onnx(model, onnx_filename, opt.batch_size_onnx, opt.dynamic)
    return


def export_onnx_qat(model, onnx_filename, batch_onnx=1, dynamic=False, simplify=True, opset=13):
    model.eval()

    # We have to shift to pytorch's fake quant ops before exporting the model to ONNX
    quant_nn.TensorQuantizer.use_fb_fake_quant = True

    # Export ONNX for multiple batch sizes
    print("Creating ONNX file: " + onnx_filename)
    dummy_input = torch.randn(batch_onnx, 3, 640, 640)     #TODO: switch input dims by model

    # YOLOv5 ONNX export    
    import onnx
    
    output_names = ['output0']
    if dynamic:
        dynamic = {'images': {0: 'batch', 2: 'height', 3: 'width'}}  # shape(1,3,640,640)        
        dynamic['output0'] = {0: 'batch', 1: 'anchors'}  # shape(1,25200,85)

    torch.onnx.export(
        model.cpu(),
        dummy_input.cpu(),
        onnx_filename,
        verbose=False,
        opset_version=opset,
        do_constant_folding=True,  # WARNING: DNN inference with torch>=1.12 may require do_constant_folding=False
        input_names=['images'],
        output_names=output_names,
        dynamic_axes=dynamic or None)

    # Checks
    model_onnx = onnx.load(onnx_filename)  # load onnx model
    onnx.checker.check_model(model_onnx)  # check onnx model
    print('ONNX export success, saved as %s' % onnx_filename)    

    # Simplify
    if simplify:
        try:
            import onnxsim
            model_onnx, check = onnxsim.simplify(model_onnx)
            assert check, 'assert check failed'
            onnx.save(model_onnx, onnx_filename)
        except Exception as e:
            print('ONNX simplifier failure')            

    # Restore the PSX/TensorRT's fake quant mechanism
    quant_nn.TensorQuantizer.use_fb_fake_quant = False

    return True
