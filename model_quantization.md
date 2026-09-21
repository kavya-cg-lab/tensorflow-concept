Model Quantization - Interview Standard Explanation
1. Standard Definition

Model Quantization is the process of converting a neural network's high-precision numerical values (such as FP32 floating-point numbers) into lower-precision representations (such as INT8, INT4, or FP16) to reduce model size, memory consumption, power usage, and inference latency while maintaining acceptable accuracy.

Interview Definition

"Quantization is a model optimization technique in which floating-point weights and activations are converted into lower-precision data types such as INT8 or INT4. This reduces memory usage and computation cost, enabling faster inference on edge devices, mobile phones, DSPs, NPUs, and embedded systems with minimal accuracy loss."

2. Why Quantization Is Needed?

Assume a trained model contains millions of weights.

FP32 Model
Weight1 = 0.234567
Weight2 = -1.567890
Weight3 = 2.345678


Each value occupies:

32 bits = 4 bytes


For a model with:

100 million parameters


Memory:

100M × 4 bytes
= 400 MB


This creates:

High memory usage
High power consumption
Higher latency
Difficult deployment on mobile devices

Therefore we convert:

FP32 → INT8


Result:

4× smaller model
Faster inference
Less memory usage
Lower power consumption
3. Understanding Number Representations
Floating Point Numbers

These can store decimal values.

Examples:

3.14159
0.125
-2.75
45.678


Used during training because they provide high precision.

Structure of Floating Point

FP32 contains:

Sign Bit
Exponent
Mantissa


Example:

3.14


Computer internally stores:

Sign = Positive
Exponent = Scale
Mantissa = Precision digits

Memory
FP32 = 32 bits
FP16 = 16 bits

Integer Numbers

Integers contain no decimal values.

Examples:

5
10
-7
120

INT8 Range
-128 to 127


Requires:

8 bits


Only:

1 byte


This makes computation much faster.

4. Floating Point vs Integer
Feature	Floating Point	IntegerDecimal Support	Yes	No
Precision	High	Lower
Memory	High	Low
Speed	Slower	Faster
Training	Mostly Used	Rare
Inference	Used	Common
Interview Answer

Floating-point numbers provide high precision and are generally used during training, whereas integers require less memory and computation, making them suitable for inference on resource-constrained devices.

5. What is Fixed Point Representation?

This is commonly asked in embedded AI interviews.

Suppose:

1.23


Instead of storing:

1.23


Store:

123


and remember:

Decimal Position = 2


Thus:

123 → 1.23


This is called Fixed Point Representation.

Difference

Floating Point:

Decimal can move


Fixed Point:

Decimal position fixed


Used in:

Embedded systems
DSP processors
Low-power AI devices
6. Core Idea Behind Quantization

Assume model weights are:

0.1
0.3
0.5
0.7
1.0


INT8 supports:

-128 to 127


We convert each floating-point value to an integer.

Example:

0.1 → 13
0.3 → 38
0.5 → 64
0.7 → 89
1.0 → 127


Now all weights become integers.

This mapping process is called:

Quantization

7. Important Terminology
Weights

Parameters learned during training.

Example:

W = [0.23, -1.45, 3.2]

Activations

Output produced by each layer.

Example:

Input → Layer → Activation


Example activation:

2.45
-0.78
1.92


Both weights and activations can be quantized.

Calibration

Process of collecting activation ranges before quantization.

Example:

Minimum = -10
Maximum = 12


Used to calculate scale factors.

8. Scale and Zero Point

Most important interview topic.

Because:

Float Values


and

INT8 Values


have different ranges.

A mapping rule is required.

Scale

Represents how much one quantized step corresponds to in floating-point space.

Formula:

Scale =
(MaxFloat - MinFloat)
/
(MaxInt - MinInt)

Example

Float range:

0 to 1


INT8 range:

0 to 255


Scale:

1 / 255
= 0.00392

Zero Point

Integer value representing real value zero.

Used in asymmetric quantization.

Quantization Formula
Q = round(R / Scale) + ZeroPoint


Where:

Q = Quantized Integer
R = Real Float Value

Example
Float Value = 0.5
Scale = 0.00392
Zero Point = 0


Then:

Q = round(0.5 / 0.00392)

Q ≈ 128


Hence:

0.5 → 128

Dequantization

Recover float again.

Formula:

RealValue =
(Q - ZeroPoint) × Scale


Example:

128 × 0.00392

≈ 0.5

9. Quantization Workflow
Step 1: Train Model

Train normally in FP32.

Dataset
   ↓
Training
   ↓
FP32 Model

Step 2: Analyze Range

Find:

Minimum value
Maximum value


for weights and activations.

Example:

Min = -5
Max = 5

Step 3: Compute Scale & Zero Point
Scale
Zero Point


are calculated.

Step 4: Convert FP32 to INT8

Example:

0.5 → 13
1.5 → 38
2.8 → 71

Step 5: Save Quantized Model
FP32 Model = 100 MB

INT8 Model = 25 MB

Step 6: Deploy

Run inference on:

CPU
GPU
DSP
NPU
Edge Device
10. Types of Quantization
1. Dynamic Quantization

Only weights are quantized.

Weights → INT8
Activations → FP32

Advantages
Easy
Fast implementation
Small accuracy drop
Interview Answer

Dynamic quantization quantizes only the model weights while activations are quantized dynamically during inference.

2. Static Quantization

Both weights and activations are quantized.

Weights → INT8
Activations → INT8


Requires:

Calibration Dataset


Advantages:

Better performance

3. Quantization Aware Training (QAT)

Quantization effects are simulated during training.

Training
   ↓
Fake Quantization
   ↓
Model learns quantization errors
   ↓
Deployment

Advantages

Highest accuracy.

Interview Answer

QAT simulates quantization during training so that the network learns to compensate for quantization errors, resulting in better accuracy after deployment.

11. PTQ vs QAT
PTQ (Post Training Quantization)
Train Model
      ↓
Quantize
      ↓
Deploy


Advantages:

Easy
No retraining

Disadvantages:

Accuracy loss possible
QAT
Train
 ↓
Quantization Simulation
 ↓
Retrain
 ↓
Deploy


Advantages:

Best accuracy

Disadvantages:

More training time
12. Symmetric vs Asymmetric Quantization
Symmetric Quantization

Range:

-127 to 127


Zero remains:

0


Formula:

Q = round(Float/Scale)


Simple and efficient.

Usually used for weights.

Asymmetric Quantization

Range:

0 to 255


Uses:

Zero Point


Formula:

Q = round(Float/Scale) + ZeroPoint


Usually used for activations.

13. Per-Tensor vs Per-Channel Quantization
Per-Tensor

Single scale for entire tensor.

Tensor
 ↓
One Scale


Fast but may lose accuracy.

Per-Channel

Each channel has its own scale.

Channel1 → Scale1
Channel2 → Scale2
Channel3 → Scale3


More accurate.

Widely used in:

CNNs
LLM deployments
Qualcomm SNPE
TensorRT
TFLite
14. Accuracy Trade-Off

Example:

FP32 Accuracy = 95%


After INT8:

94.8%


Very little loss.

After INT4:

93%


Accuracy usually drops more.

Trade-off:

More Compression
      ↓
Less Memory
      ↓
Potential Accuracy Loss

15. Quantization in LLMs

Suppose:

7 Billion Parameters

FP32
7B × 4 bytes

≈ 28 GB

INT8
7B × 1 byte

≈ 7 GB

INT4
7B × 0.5 byte

≈ 3.5 GB


This is why:

Llama
Phi
Gemma
Qwen

can run on mobile and edge devices.

16. Real Example

Suppose a layer weights are:

[0.2, 0.5, 0.8, 1.0]


Range:

Min = 0
Max = 1


Scale:

1 / 255
≈ 0.00392


Quantized values:

0.2 → 51
0.5 →128
0.8 →204
1.0 →255


Stored as:

[51,128,204,255]


instead of floating point values.

Memory reduced by approximately 4×.

Perfect Interview Answer (2 Minutes)

"Model quantization is an optimization technique that converts high-precision floating-point weights and activations, such as FP32, into lower-precision representations like INT8, INT4, or FP16. The main goal is to reduce model size, memory usage, power consumption, and inference latency while maintaining acceptable accuracy. Quantization works by mapping floating-point values into integer values using parameters called Scale and Zero Point. The typical workflow is: train the model in FP32, collect min-max ranges, calculate scale and zero-point values, convert weights and activations into INT8 or INT4, and then deploy the optimized model. Common approaches include Post Training Quantization (PTQ), Dynamic Quantization, Static Quantization, and Quantization Aware Training (QAT). QAT generally provides the best accuracy because the model learns quantization effects during training. Quantization is widely used for deploying AI models on mobile phones, DSPs, NPUs, edge devices, and embedded systems."

One-Line Interview Summary

"Quantization is the process of converting FP32 model parameters into lower-precision formats like INT8 or INT4 using scale and zero-point mapping to achieve smaller model size and faster inference with minimal accuracy loss."
