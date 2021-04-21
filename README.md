
# summary 
Neural networks can learn well even if the data is skewed, unusually scaled, or otherwise non-optimal. 
However, it may be difficult to learn if the sample is not large enough for the Law of Large Numbers to work. 

For demonstration (and play), various transformation algorithms are applied here to a synthetic dataset to observe the effects on neural network analysis. 

## synthetic data, synthetic hurdles
Create a synthetic classification dataset with (default) 5 features. 
Method: sklearn.datasets.make_classification
The creation of the data is based on a normal distribution. 

One goal is to mimic real data problems. Therefore intentionally create hurdles by including shifting, noise, and restricting the sample and the features available for learning. 

## analysis
* cross validataion
* train-test-split
* transformation: StandardScaler, Normalizer, QuantileTransformer(norm)
* keras neural network

## results


![F1_unscaled_loss](https://user-images.githubusercontent.com/82636544/115548186-7ae50680-a2a7-11eb-8d16-9a74c05bccb6.png)

![F2_unscaled_acc](https://user-images.githubusercontent.com/82636544/115548190-7b7d9d00-a2a7-11eb-83e3-bc0224b6dee5.png)

![F3_Standard_loss](https://user-images.githubusercontent.com/82636544/115548192-7c163380-a2a7-11eb-8be4-ccd79aa65304.png)

![F4_Standard_acc](https://user-images.githubusercontent.com/82636544/115551534-86d2c780-a2ab-11eb-95d7-890997cf3a72.png)

![F5_Normalizer_loss](https://user-images.githubusercontent.com/82636544/115548198-7caeca00-a2a7-11eb-8d7e-a1c3beaf181e.png)

![F6_Normalizer_acc](https://user-images.githubusercontent.com/82636544/115548200-7caeca00-a2a7-11eb-8e04-db85012cd3a1.png)

![F7_Quantile_norm_loss](https://user-images.githubusercontent.com/82636544/115548201-7d476080-a2a7-11eb-901a-012af1d36766.png)

![F8_Quantile_norm_acc](https://user-images.githubusercontent.com/82636544/115548202-7d476080-a2a7-11eb-8955-fe5c6c64543d.png)
