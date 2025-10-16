### Time complexity: rough estimates

> Data from performance mode, plugged in.
> - CPU: AMD Ryzen 9 9955HX
> - GPU: Nvidia RTX 5070Ti


| System size $N$ | CUDA - GPU (s) | CPP - CPU (s) | Speedup factor |
| -------- | ------- | ------ | -------------- |
| 256   | 1.305   | 2.990   | 2.29  |
| 512   | 1.952   | 11.345  | 5.81  |
| 1024  | 3.405   | 46.681  | 13.71 |
| 2048  | 5.893   | 182.104 | 30.89 |
| 4096  | 12.255  | 745.358 | 60.85 |
| 8172  | 28.683  |         |       |
| 16384 | 89.961  |         | $\approx$ 138 (estimated) |      |

CPP scales with $\mathcal{O}(t^2)$, as expected, since we're coupling each element to each other element (though $\mathbb{A}$) every step.