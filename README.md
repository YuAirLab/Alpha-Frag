# Alpha-Frag

Alpha-Frag is a deep neural network to predict the presence of peptide fragment ions. You can use it directly without installation.

## Hardware

A [GPU with CUDA support](https://developer.nvidia.com/cuda-gpus)

## Package

- [PyTorch](https://pytorch.org/get-started/locally/#windows-anaconda) 1.0.0+

## Example

1. Sequences with amino acid [BJOUXZ] are not supported.
2. Each C is treated as Cysteine with carbamidomethylation.
3. Modifications except "M(ox)" are not supported.
M(ox) should be written as 'm'.
4. The length of peptide should be between 7 and 30.
5. The precursor charge should be between 1 and 4.
6. Output such as 'y10_2' means 'y' type ions and cleavage size at 10 with charge 2. More than 2 charges fragment ion is not considered.

- run by case:
```shell script
python run_by_case.py ACDEFGHIKLMmNPQRSTVWYK 2
```
output:
```shell script
ACDEFGHIKLMmNPQRSTVWYK_2: y1_1;y2_1;y3_1;y8_1;y9_1;y10_1;y11_1;y12_1;y13_1;y14_1;y15_1;b2_1;b3_1;b4_1;b7_1;b8_1;b9_1;b10_1
```

- run by DataFrame:

The input DataFrame should include 'simple_seq' and 'pr_charge' columns.

see [run_by_df.py](https://github.com/YuAirLab/Alpha-Frag/blob/master/run_by_df.py)



