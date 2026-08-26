# `task3_v2`

Evaluation of Task 3 across a few axes:
- model checkpoint (`pt-full`, `walnut`)
- feature depth (16, final i.e. 24)
- augmentation (train, test, both, none)

## Results

| ckpt | depth | train aug | test aug | r | 95% CI | MAE | 95% CI | camcan r | camcan MAE | alpha | ‖w‖ | time |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pt-full | final | False | False | **0.963** | 0.957 – 0.969 | **3.69** | 3.45 – 3.95 | 0.947 | 5.06 | 100 | 4.958 | 38s |
| pt-full | final | True | False | **0.958** | 0.952 – 0.965 | **3.97** | 3.70 – 4.25 | 0.947 | 5.79 | 100 | 6.854 | 10s |
| pt-full | final | False | True | **0.950** | 0.942 – 0.958 | **8.23** | 7.74 – 8.75 | 0.929 | 8.65 | 100 | 4.958 | 4s |
| pt-full | final | True | True | **0.955** | 0.948 – 0.962 | **4.20** | 3.92 – 4.49 | 0.945 | 5.67 | 100 | 6.854 | 12s |
| pt-full | 16 | False | False | **0.967** | 0.962 – 0.971 | **3.55** | 3.32 – 3.80 | 0.942 | 5.53 | 100 | 4.570 | 4s |
| pt-full | 16 | True | False | **0.960** | 0.953 – 0.965 | **3.90** | 3.63 – 4.17 | 0.940 | 5.49 | 100 | 5.971 | 9s |
| pt-full | 16 | False | True | **0.951** | 0.944 – 0.959 | **5.94** | 5.56 – 6.35 | 0.936 | 6.55 | 100 | 4.570 | 4s |
| pt-full | 16 | True | True | **0.955** | 0.948 – 0.962 | **4.15** | 3.87 – 4.43 | 0.943 | 5.50 | 100 | 5.971 | 9s |
| walnut-vitl | final | False | False | **0.968** | 0.963 – 0.972 | **3.50** | 3.29 – 3.74 | 0.945 | 5.65 | 100 | 4.750 | 4s |
| walnut-vitl | final | True | False | **0.964** | 0.958 – 0.969 | **3.68** | 3.43 – 3.95 | 0.946 | 6.08 | 100 | 6.354 | 9s |
| walnut-vitl | final | False | True | **0.959** | 0.953 – 0.965 | **4.36** | 4.09 – 4.65 | 0.944 | 6.24 | 100 | 4.750 | 4s |
| walnut-vitl | final | True | True | **0.961** | 0.955 – 0.966 | **3.91** | 3.65 – 4.18 | 0.949 | 5.80 | 100 | 6.354 | 9s |
| walnut-vitl | 16 | False | False | **0.967** | 0.962 – 0.972 | **3.49** | 3.26 – 3.73 | 0.951 | 5.44 | 316.2 | 2.765 | 4s |
| walnut-vitl | 16 | True | False | **0.961** | 0.956 – 0.966 | **3.85** | 3.60 – 4.12 | 0.947 | 5.79 | 316.2 | 3.871 | 10s |
| walnut-vitl | 16 | False | True | **0.959** | 0.953 – 0.965 | **4.46** | 4.17 – 4.77 | 0.950 | 6.09 | 316.2 | 2.765 | 4s |
| walnut-vitl | 16 | True | True | **0.959** | 0.953 – 0.964 | **4.05** | 3.79 – 4.33 | 0.950 | 5.69 | 316.2 | 3.871 | 10s |

Observations (from CL):

- performance on CamCAN much improved now that we apply synthseg. basically no generalization gap.
- training with augmentation helps performance on test-time augmentation.
- both depths about the same
- both checkpoints about the same
