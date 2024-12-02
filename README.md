# Replicating article: Human Stress Detection With Wearable Sensors Using Convolutional Neural Networks

### You can either 

1. execute <code><b>create_data.py</b></code> script to just create data needed for replication, or
2. execute <code><b>run_all_experiments.py</b></code> to just execute experiments from the paper, assuming you already have the data created, or
3. execute <code><b>main.py</b></code> script to create data and run all the experiments all in one file. This might take about 16+ hours.

### Lengths of execution:
- creating data approximately 3 hours,
- running all experiments (9 experiments) approximately 13,5 hours

---

## Changes in Methodology Compared to the Original Article

### 1st Change
RespiBAN (chest) data were resampled from 700 Hz to 64 Hz, matching the sampling frequency of the Empatica E4 (wrist) device. This step was not performed in the original replicated article.

---

### 2nd Change

**In the replicated article, three types of classifications were performed:**

1. Differentiating between stress (stress) and non-stress states (baseline and amusement phases) **(S vs NS)**.

2. Differentiating between baseline, stress, and amusement phases **(B vs S vs A)**.

3. Differentiating between five classes: baseline, stress, amusement, meditation, and recovery phases **(B vs S vs A vs M vs R)**.

The change implemented here replaces the third type of classification with differentiation between only four states. This change was made because the dataset documentation made no mention of a recovery phase. According to the dataset documentation, the states (data annotations) are described as follows, and we adhered to these definitions during data processing:  
*0 = not defined / transient, 1 = baseline, 2 = stress, 3 = amusement, 4 = meditation, 5/6/7 = should be ignored in this dataset.*

Based on this documentation:
- Data annotated with values 5, 6, or 7 were removed. 
- A considerable amount of data was annotated with the value 0, which, according to the documentation, means "not defined/transient." These could represent anomalies or transitional states between phases. The replicated article does not mention how these data were handled, so we decided to remove them. 
- Consequently, the final dataset only included data annotated with the values 1, 2, 3, and 4.

---

### 3rd Change
The replicated article does not mention whether data normalization or standardization was performed. However, it is likely that normalization or standardization was applied, as it facilitates the convergence of neural networks, a fact the article's authors were likely aware of. In our approach, we used `StandardScaler()` from the scikit-learn library.

---

## Results (replicated):
<table>
  <caption>Table 2. Accuracy (%) and F1-score (%) Depending on the Signal Processing Module Considering all Signals in the Three Classification Tasks</caption>
  <thead>
    <tr>
      <th rowspan="2">Signal processing</th>
      <th colspan="2">S vs NS</th>
      <th colspan="2">B vs S vs A</th>
      <th colspan="2">B vs S vs A vs M</th>
    </tr>
    <tr>
      <th>Accuracy</th>
      <th>F1-score</th>
      <th>Accuracy</th>
      <th>F1-score</th>
      <th>Accuracy</th>
      <th>F1-score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Fourier transform</td>
      <td>81,78 ± 0,65</td>
      <td>80,44 ± 0,65</td>
      <td>74,15 ± 0,76</td>
      <td>62,77 ± 0,84</td>
      <td>73,03 ± 0,78</td>
      <td>59,86 ± 0,87</td>
    </tr>
    <tr>
      <td>Fourier + Cube Root</td>
      <td>85,92 ± 0,56</td>
      <td>84,38 ± 0,56</td>
      <td>75,68 ± 0,84</td>
      <td>62,99 ± 0,83</td>
      <td>76,36 ± 0,74</td>
      <td>65,32 ± 0,82</td>
    </tr>
    <tr>
      <td>Fourier + Cube Root + CQT</td>
      <td>77,93 ± 0,71</td>
      <td>76,03 ± 0,71</td>
      <td>65,84 ± 0,83</td>
      <td>53,40 ± 0,87</td>
      <td>65,53 ± 0,84</td>
      <td>51,97 ± 0,88</td>
    </tr>
  </tbody>
</table>

## Results (original):
<table>
  <caption>Table 2. Accuracy (%) and F1-score (%) Depending on the Signal Processing Module Considering all Signals in the Three Classification Tasks</caption>
  <thead>
    <tr>
      <th rowspan="2">Signal processing</th>
      <th colspan="2">S versus NS</th>
      <th colspan="2">B versus S versus A</th>
      <th colspan="2">B versus S versus A versus M</th>
    </tr>
    <tr>
      <th>Accuracy</th>
      <th>F1-score</th>
      <th>Accuracy</th>
      <th>F1-score</th>
      <th>Accuracy</th>
      <th>F1-score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Fourier transform</td>
      <td>95,20 ± 0,13</td>
      <td>95,13 ± 0,13</td>
      <td>83,12 ± 0,13</td>
      <td>82,67 ± 0,13</td>
      <td>79,67 ± 0,25</td>
      <td>79,24 ± 0,25</td>
    </tr>
    <tr>
      <td>Fourier + Cube Root</td>
      <td>96,73 ± 0,11</td>
      <td>96,65 ± 0,11</td>
      <td>85,00 ± 0,11</td>
      <td>84,92 ± 0,22</td>
      <td>81,21 ± 0,24</td>
      <td>81,45 ± 0,24</td>
    </tr>
    <tr>
      <td>Fourier + Cube Root + CQT</td>
      <td>96,62 ± 0,11</td>
      <td>96,63 ± 0,11</td>
      <td>85,03 ± 0,22</td>
      <td>85,01 ± 0,22</td>
      <td>81,15 ± 0,24</td>
      <td>81,70 ± 0,24</td>
    </tr>
  </tbody>
</table>

---

## Reference:
Manuel Gil-Martin, Ruben San-Segundo, Ana Mateos, and Javier Ferreiros-Lopez. (2022). Human Stress Detection With Wearable Sensors Using Convolutional Neural Networks. Ciudad Universitaria, 28040 Madrid, Spain. DOI: 10.1109/MAES.2021.3115198. 
