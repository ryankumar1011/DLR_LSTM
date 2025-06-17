Dataset:
The dataset is from Samsung INR21700 30T 3Ah Li-ion Battery Data taken by Phillip Kollmeyer
The battery used is an NMC rechargeable round cell
https://data.mendeley.com/datasets/ycx459r5c3/2

RELEVANT CODE SOURCES:
    https://github.com/KeiLongW/battery-state-estimation/tree/main
    https://github.com/ArpanBiswas99/Battery-State-of-Charge-Estimation/tree/main
    https://www.kaggle.com/code/aditya9790/soc-estimation-lstm/notebook

Main project:
We want to model battery performance in dynamic operating conditions
In practical applications, the battery is dynamically charged and discharged
We also cannot measure internal resistance, since that requires sophisticated equipment in a controlled environment
Using supervised models means that accuracy of the models depends on historical data samples
But:
    We don't want any large errors when the battery cells are inconsistent
    We don't know why the battery will fail. We won't be able to correctly model early degradation
    The model could be used on different battery types. We want to model to generalize well on new data

We want to use a block-box NN model to add non-linearity and allow the model to adapt to noisy data

This project:
We will use dataset with SOC already given
The models used are LSTMs and Random Forests

Current Goal: prototype EDA, data cleaning, visualization, and implementing NN model
End Goal: Develop an Unsupervised model to model SOC given a power profile

About dataset used:
The paper I'm following used battery data on Li-ion batteries tested under 8 different known drive cycle patterns
This dataset uses four different drive cycle patterns instead (UDDS, HWFET, LA92, US06)
It also has data on eight drive cycles are made of random mixes of the above 4

Important : The tests are power profile is calculated for a single LG HG2 cell (a Li-ion cell). This might not be best
when modelling battery modules as a whole

The tests are performed at temperatures 40degC, 25degC, 10degC, 0degC, -10degC, and -20degC

Multiple different types of tests are performed (HPPC test at different SOCs, Charging and Discharging at different C-rates,
Drive cycles depending on different power profiles, etc)

The folder contain both csv files and .mat files. Meta information is given at top 30 lines of the csv files.