# Outputs + Images
TensorFlow version: 2.19.0 |
(50000, 32, 32, 3) (50000, 1) (10000, 32, 32, 3) (10000, 1)

<img width="543" height="417" alt="image" src="https://github.com/user-attachments/assets/18a7f710-8081-483b-97d4-f530ee8190ac" />

Number of classes: 10 |
Model: "functional_1"
| Layer (type)               | Output Shape        | Param #   |
|-----------------------------|---------------------|-----------|
| InputLayer                 | (None, 32, 32, 3)   | 0         |
| Conv2D                     | (None, 32, 32, 32)  | 896       |
| BatchNormalization          | (None, 32, 32, 32)  | 128       |
| Conv2D                     | (None, 32, 32, 32)  | 9,248     |
| BatchNormalization          | (None, 32, 32, 32)  | 128       |
| MaxPooling2D               | (None, 16, 16, 32)  | 0         |
| Conv2D                     | (None, 16, 16, 64)  | 18,496    |
| BatchNormalization          | (None, 16, 16, 64)  | 256       |
| Conv2D                     | (None, 16, 16, 64)  | 36,928    |
| BatchNormalization          | (None, 16, 16, 64)  | 256       |
| MaxPooling2D               | (None, 8, 8, 64)    | 0         |
| Conv2D                     | (None, 8, 8, 128)   | 73,856    |
| BatchNormalization          | (None, 8, 8, 128)   | 512       |
| Conv2D                     | (None, 8, 8, 128)   | 147,584   |
| BatchNormalization          | (None, 8, 8, 128)   | 512       |
| MaxPooling2D               | (None, 4, 4, 128)   | 0         |
| Flatten                    | (None, 2048)        | 0         |
| Dropout                    | (None, 2048)        | 0         |
| Dense                      | (None, 1024)        | 2,098,176 |
| Dropout                    | (None, 1024)        | 0         |
| Dense                      | (None, 10)          | 10,250    |

Total params: 2,397,226 (9.14 MB)  
Trainable params: 2,396,330 (9.14 MB)  
Non-trainable params: 896 (3.50 KB)
