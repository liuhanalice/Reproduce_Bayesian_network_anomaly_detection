# Reproduce Report

## EM_options for generating point cloud 

```
class EM_options(
    bg_k: float,
    outliers_rate: float,
    defect_depth: int,
    defect_radius: float,
    trans: float,
    defect_pos: ndarray,
    spline_paras: Unknown | None = None,
    spline_knot: Unknown | None = None,
    bg_size: float = 20,
    bg_std_depth: float = 0.15,
    bg_std_xy: float = 0.02,
    outliers_std_depth: float = 1.5,
    outliers_std_xy: float = 0.02,
    step: float = -0.45,
    spline_flag: bool = True
)
```

Single_Experiment setting (for depth 3):
``` 
options = EM_options(0.0004, 0.01, 3, 2.4, 1.5, defect_pos=np.array([[-0.0], [-0.0]]), bg_std_depth=0.10, step=-0.4, spline_flag=False) 
```
![alt text](image.png)
- bg_k: reference surface shape
  - alternate example: 0.001
  ![alt text](image-1.png)
- outliers_rate: number of outliers
  - alternate example: 0.1
  ![alt text](image-2.png)
- defect_depth, defetct_radius, trans: see below image (cropped from the paper)
![alt text](image-3.png)

defect_depth: $d$

defect_radius: $r_2$

trans: $\tau = \frac{r_2}{r_1}$, steepness, smaller --> steeper

- defect_pos: defect position (x,y)
  - alternate example: (-0.9, -1.4)
  ![alt text](image-4.png)

- bg_std_depth: different noise variance, refer to experiment 1 in 5.1.1 in the paper
  - alternate example: 0.15
  ![alt text](image-5.png)

- step: <0, absolute value = the gap between surface and red defect area (mentioned in the top left parts of the paper on page 1182 and Figure 6), note that the blue defact part does not move down accordingly
![alt text](image-6.png)


## Experiments

### Repeat
|Experiment d=3 | FPR   | FNR |
| -------- | ------- | ------- |
| original (30 experiments average)  | 1.07761704e-05   | 0.0851115 |
| 5 experiments average   |  1.8478311082367072e-05   | 0.05334157015366224|
| single experiment | 0.0   | 0.03213957759412305 |



|Experiment d=7 | FPR   | FNR |
| -------- | ------- | ------- |
| original (30 experiments average)  | 0.00407483847  |  0.01899044 |
| 5 experiment average   |  0.0033792474280080684  | 0.018663362763552894 |
| single experiment| 0.004533243787777031    | 0.01137521222410866|

### Generate Defect on Corners
Depth = 3, Single Experiment:
```
    options = EM_options(0.0008, 0.01, 3, 2.4, 1.5, defect_pos=np.array([[-0.9], [-1.4]]), bg_std_depth=0.1, step=-0.35, spline_flag=False)
```
![alt text](image-7.png)
![alt text](image-8.png)