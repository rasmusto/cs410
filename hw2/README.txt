1. See other attachments.
2.
    n = 3, m = 3
    cuda: 21188us
    sequential 224us
    pthreads 440us
    openmp 239us
3.
    1. CUDA was by far the slowest (because it had the most overhead).  Sequential, pthreads, and openmp were all comparable with this small of a dataset.
    2. Openmp was the easiest to code by far.  It didn't have to change from the sequential version, and silently implemented parallelism.
    3. CUDA was the most difficult to program because it required manual management of the host and device memories.
    4. I chose a small matrix to see how large a factor overhead was on execution time.  Also, I ran into segmentation faults if I tried numbers that were too large.
