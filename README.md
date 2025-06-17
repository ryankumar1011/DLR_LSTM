
# Battery State of Charge (SOC) Estimation using LSTM

This is a stacked LSTM SOC (State of Charge) Estimation model I built while interning at The German Aerospace Center (DLR).

This project models battery performance in dynamic operating conditions using drive cycle test data from multiple open-source datasets. The implementation uses a black-box neural network approach with LSTM models to add non-linearity and adapt to noisy data, making it well-suited for forecasting time-series data like SOC.

## Project Overview

The main objective is to develop a robust SOC estimation model that can handle dynamic charging and discharging conditions in practical applications. The project includes comprehensive data exploration, cleaning, visualization, and LSTM model implementation.

### Key Features
- **Multi-dataset support**: Handles both Samsung INR and CALB battery datasets
- **LSTM-based forecasting**: Optimized for time-series SOC prediction
- **Dynamic operating conditions**: Trained on various drive cycle patterns
- **Temperature variance**: Models performance across different temperature conditions
- **Flexible data processing**: Configurable preprocessing pipeline

## Datasets

### Primary Features
- **Voltage**: Battery terminal voltage
- **Current**: Charging/discharging current
- **Temperature**: Operating temperature
- **Label**: State of Charge (SOC)

### INR Dataset
**Samsung INR21700 30T 3Ah Li-ion Battery Data** by Phillip Kollmeyer
- **Cell Type**: NMC rechargeable round cell (single LG HG2 cell)
- **Link**: https://data.mendeley.com/datasets/9xyvy2njj3/1
- **Temperature Range**: -20°C to 40°C (40°C, 25°C, 10°C, 0°C, -10°C, -20°C)
- **Test Types**: 
  - HPPC tests at different SOCs
  - Fixed C-rate charge/discharge tests
  - Drive cycle tests (UDDS, HWFET, LA92, US06)
  - 8 tests from random mix of drive cycle patterns
- **Format**: CSV files with metadata in top 30 lines

### CALB Dataset
**CALB L148N58A Large Prismatic Li-ion Battery Data**
- **Cell Type**: NMC Large Prismatic
- **Link**: https://data.mendeley.com/datasets/cp3473x7xv/3
- **Cells Tested**: 11 different prismatic cells
- **Temperature Range**: 10°C, 25°C, 40°C
- **Test Types**:
  - HPPC tests
  - Drive cycle tests (WLTP, UDDS, US06)
  - EIS tests
- **Format**: Excel files

## Architecture

### LSTM Model (`SocLSTM` class)
- **Input Shape**: `(batch_size, sequence_length, input_size)`
- **Output Shape**: `(batch_size, sequence_length, output_size)`
- **Activation Function**: SELU
- **Loss Function**: `nn.SmoothL1Loss()` (PyTorch HuberLoss implementation)
- **Architecture**: 3 fully connected layers (fc1, fc2, fc3)
- **Output**: Hidden states and final state (not single value)

### Data Processing Pipeline
The `get_data()` function in `data_extraction.py` provides flexible data retrieval:

```python
get_data(file_paths, read_file, min_max_path=None, step_size=None, 
         normalize=False, smoothen=False, window_size=1, update_min_max=False)
```

**Parameters:**
- `file_paths`: List of relative file paths from project root
- `read_file`: Function returning (features: DataFrame, label: DataFrame)
- `min_max_path`: Numpy file path for normalization parameters
- `step_size`: Breaks data into independent chunks for stateless LSTM
- `window_size`/`smoothen`: Moving window smoothing parameters
- `update_min_max`: Flag to update normalization file

## Project Structure

```
├── data_extraction.py      # Data loading and preprocessing
├── experiment.py          # Training configuration and execution
├── models/                # LSTM model implementations
├── experiments/           # Experiment results and logs
│   ├── [experiment_name]/
│   │   ├── trained_model.pth
│   │   ├── experiment_log.txt
│   │   ├── min_max_features.npy
│   │   └── plots/
└── datasets/             # Raw data files
```

## Usage

### Running Experiments
Configure and run experiments using `experiment.py`:
- Modify experiment parameters (data/model/training hyperparameters)
- Change experiment name to create new experiment folder
- Results automatically saved with logs, model weights, and plots

### Dataset Integration
Use the `ds` class to manage different datasets:
```python
# Example usage
inr_dataset = ds(read_file=read_inr, file_paths=inr_file_paths)
calb_dataset = ds(read_file=read_calb, file_paths=calb_file_paths)
```

### Model Evaluation
Use `plot_predictions()` to visualize results:
- Plots `prediction[::299]` (final values for step_size=300)
- Compares predicted vs actual SOC values

## Technical Considerations

### Memory Optimization
- `step_size` parameter breaks data into independent chunks
- Reduces memory burden on LSTM
- Enables more frequent weight updates
- Alternative: `step_size=None` pads sequences to max_length

### Data Preprocessing
- Normalization using stored min/max values
- Moving window smoothing for noise reduction
- Support for both CSV and Excel formats
- Flexible file path management with keyword-based filtering

## References

**Main Reference** (data extraction, LSTM model):
- https://www.kaggle.com/code/aditya9790/soc-estimation-lstm/notebook

**Supplementary References** (LSTM model, hyperparameters):
- https://github.com/KeiLongW/battery-state-estimation/tree/main
- https://github.com/ArpanBiswas99/Battery-State-of-Charge-Estimation/tree/main

## Future Development

### Current Goal
Prototype EDA, data cleaning, visualization, and NN model implementation

### End Goal
Develop an unsupervised model to estimate SOC given a power profile, enabling:
- Better generalization across different battery types
- Adaptation to battery degradation patterns
- Reduced dependency on historical labeled data

## Important Notes

- The INR dataset focuses on single LG HG2 cell performance, which may not directly translate to battery module behavior
- Drive cycle patterns vary between datasets (INR: UDDS, HWFET, LA92, US06; CALB: WLTP, UDDS, US06)
