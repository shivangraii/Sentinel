# Edge AI on UAV Hardware

Running neural networks onboard avoids depending on a data link and cuts latency, but UAV compute is limited by weight, size and power.

## Hardware and runtimes

The NVIDIA Jetson family and the Google Coral Edge TPU are common platforms for onboard inference. TensorRT is an inference optimizer that targets NVIDIA GPUs, while ONNX Runtime and TensorFlow Lite provide portable runtimes for edge devices. The YOLO family of detectors is widely used for real-time object detection onboard.

## Model compression

Quantization lowers the numeric precision of weights. INT8 quantization stores weights in 8 bits instead of 32, which makes the model roughly 4x smaller than FP32, and 4-bit quantization gives roughly 8x compression. There are two main approaches: post-training quantization (PTQ), which converts a trained model with little or no retraining, and quantization-aware training (QAT), which simulates low precision during training to preserve accuracy. Pruning removes redundant weights, and knowledge distillation trains a small student model to mimic a larger teacher model.

## Power and endurance

Onboard compute competes with propulsion for battery capacity. A heavier computer or a hotter GPU shortens flight time, so efficiency is a first-class design goal: smaller quantized models let the same battery support longer missions.
