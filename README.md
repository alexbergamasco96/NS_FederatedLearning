# Non-Stationary Federated Learning

Starting from the concepts of Federated Learning, we are trying to create an algorithm capable of modelling the **Time Non-Stationariety** in **Federated Learning enviroment**.
Before doing it, we assess the theoretical properties of the algorithm in standard conditions, inserting two strong hypothesis which give us the possibility to guarantee such properties:
1. Linearity of the world, seen as the phenomenon that we want to model
2. IID in the space of Federated Learning

Subsequently,  the  first  hypothesis  will  be  weakened  when  we  consider  a  non-linear preprocessing of the data, but instead the applied model will be linear.  This construction and the utilization of linear models guarantee certain theoretical properties (e.g.:  the parameters we want to estimate are Gaussian distributed, with unknown mean and covariance but we can estimate it during time).  It is important to start analysing a linear world, and then introducing a partial non-linearity.  Linear models that interest us are linear multivariate regression models and I/O linear models for time series.
