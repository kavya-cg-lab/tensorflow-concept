# CPU (Central Processing Unit)
A CPU is the “brain” of a computer that handles most general tasks like running your operating system, browsing, or basic computations. It has a few cores (usually 4–16 in consumer computers) optimized for sequential tasks and flexibility. For example, when rendering a picture in software, the CPU can run logic, load textures, and manage tasks that require varied instructions, but it may take longer for highly repetitive tasks compared to GPUs.
## CPU Memory
Central Processing Units (CPUs) rely on a hierarchical memory system:
Registers: Small, ultra-fast storage within the CPU itself for immediate computation.
Cache Memory: Multi-level (L1, L2, and sometimes L3) memory integrated on the CPU chip to reduce latency when accessing frequently used data.
System RAM: The CPU accesses main memory (DRAM) over the memory bus, which is slower than cache but much larger than on-chip memory. CPUs do not have large dedicated memory separate from system RAM; they use main RAM shared with other system components.

# GPU (Graphics Processing Unit)
A GPU contains hundreds or thousands of smaller cores designed to handle many tasks in parallel. This makes them excellent for rendering graphics, running video games, or performing tasks like image processing and simulations. For instance, 3D rendering uses GPUs to process the colors, textures, and lighting of millions of pixels simultaneously, making it much faster than a CPU alone. GPUs are also widely used in AI training because neural networks involve many identical calculations.
## Graphics Processing Units (GPUs) operate differently depending on type:
Discrete GPUs: Have dedicated memory called VRAM (Video RAM), commonly GDDR6 or HBM variants. This memory is located on the GPU card itself and is optimized for high bandwidth and parallel data access, crucial for rendering graphics and performing parallel computations in tasks like AI and simulations. VRAM is separate from system RAM, and the GPU cannot directly use CPU RAM without specific transfers over the PCIe bus.
Integrated GPUs: Some GPUs are part of the CPU (like Intel Iris or AMD APUs) and share system RAM with the CPU, as they do not have dedicated VRAM. Performance depends on system memory speed and bandwidth.

# TPU (Tensor Processing Unit)
A TPU is a specialized processor developed by Google for machine learning tasks, especially tensor operations in neural networks. Tensor operations involve large-scale matrix multiplications, which TPUs handle extremely efficiently. While CPUs and GPUs can also run AI models, TPUs offer faster processing for training and inference of models like those in image recognition or language understanding. TPUs are not designed for general-purpose tasks or normal graphics rendering—they are AI-focused.
