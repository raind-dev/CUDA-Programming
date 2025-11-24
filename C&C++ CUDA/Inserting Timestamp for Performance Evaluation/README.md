# CUDA Learning Notes – Day 8: Measuring Kernel Latency with CUDA Events

The goal of today’s exercise is to learn how to measure kernel execution latency using CUDA APIs —
specifically, how to insert timestamps around kernel launches to obtain accurate timing results.

The key feature we use is cudaEvent_t, along with the following related APIs:
- cudaEventCreate()

- cudaEventRecord()

- cudaEventSynchronize()

- cudaEventElapsedTime()

## Workflow Overview

We start by declaring two CUDA events:
``` cpp
cudaEvent_t start, stop;
```

After declaration, we use cudaEventCreate() to instantiate both events.
Next, we call cudaEventRecord() to mark a timestamp.
By placing these events before and after the kernel launch, we can measure exactly how long the kernel takes to execute.

Once we record the stop event, we must call:
``` cpp
cudaEventSynchronize(stop);
```

This ensures that the system waits until all device operations before the last cudaEventRecord() call have completed.

Finally, we call:
``` cpp
cudaEventElapsedTime(&ms, start, stop);
```

to obtain the elapsed time (in milliseconds, as a floating-point value) between the two timestamps.

With this setup, you can wrap any kernel — or even a sequence of kernels — and benchmark performance with precise CUDA-level timing.