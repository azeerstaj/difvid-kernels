### Benchmark Results:
`Tensors are identical.`

### Outputs

```txt
Input   [:5]: tensor([-1.1258, -1.1524, -0.2506, -0.4339,  0.8487], device='cuda:0')
Input  [-5:]: tensor([-1.0386, -1.6340,  0.5374,  1.0826, -1.7105], device='cuda:0')
----------
Output  [:5]: tensor([-0.1467, -0.1438, -0.1005, -0.1441,  0.6805], device='cuda:0')
Output [-5:]: tensor([-0.1554, -0.0837,  0.3786,  0.9314, -0.0747], device='cuda:0')
----------
Gt      [:5]: tensor([-0.1467, -0.1438, -0.1005, -0.1441,  0.6805], device='cuda:0')
Gt     [-5:]: tensor([-0.1554, -0.0837,  0.3786,  0.9314, -0.0747], device='cuda:0')
```