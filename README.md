# Cyberattack Prediction with Deep Learning  

## Overview

This project presents a deep learning-based approach for predicting cybersecurity attacks using the UNSW-NB15 dataset. The work is heavily inspired by two pivotal research papers:

1. **Conceptualisation of Cyberattack Prediction with Deep Learning.**
2. **Cybersecurity Attack Prediction: A Deep Learning Approach.**

By combining insights from these two papers, I have developed a model that employs a Bidirectional Long Short-Term Memory (BiLSTM) architecture, which surpasses various state-of-the-art (SOTA) methods in terms of accuracy.

## Inspiration

The first research paper, *Conceptualisation of Cyberattack Prediction with Deep Learning*, provided a solid foundation in understanding how deep learning can be applied to cybersecurity, emphasizing the need for accurate prediction systems that can identify threats in real-time.

The second paper, *Cybersecurity Attack Prediction: A Deep Learning Approach*, offered critical insights into leveraging LSTM networks for temporal data and highlighted the performance gains that could be achieved by using advanced techniques such as Bidirectional LSTM.

Taking inspiration from both of these works, I have extended the BiLSTM methodology and optimized it for the UNSW-NB15 dataset to predict potential cybersecurity attacks.

## Approach

### Dataset

The UNSW-NB15 dataset was chosen for this research because of its extensive collection of normal and malicious network activities, making it an ideal benchmark for testing intrusion detection systems. The dataset includes features such as protocol type, service, state, and other network traffic characteristics.

### Preprocessing

A series of preprocessing steps were applied to the dataset, including:

- **Feature Scaling:** MinMaxScaler was used to normalize features.
- **Label Encoding:** Categorical variables like protocol type, service, and state were encoded into numerical representations.
- **Windowing Technique:** Time-series data was prepared by using a sliding window of 100 records.

### Model Architecture

A Bidirectional LSTM (BiLSTM) model was implemented using the following architecture:

- **LSTM Layer:** Captures sequential patterns in the dataset.
- **Bidirectional LSTM Layer:** Enhances the learning by considering both past and future contexts.
- **Dropout Layer:** Reduces overfitting by randomly setting a fraction of inputs to zero during training.
- **Dense Layers:** Outputs the prediction using a sigmoid activation function.

The model was trained with a batch size of 32 and optimized using the Adadelta optimizer.

### Performance

After training the model, it achieved an accuracy of **97%**, outperforming various state-of-the-art models in the field of cyberattack prediction. This highlights the effectiveness of using BiLSTM for detecting anomalies and predicting attacks in network traffic data.

## Conclusion

This project demonstrates how a deep learning model based on Bidirectional LSTM can effectively predict cyberattacks with high accuracy. The model's performance shows significant improvement over traditional methods and offers a robust solution for real-world cybersecurity challenges.

By leveraging the findings from the two research papers, I was able to push the boundaries of cybersecurity attack prediction and offer an innovative solution that challenges current state-of-the-art techniques.

## Citations

- Ibor, A. E., Oladeji, F. A., Okunoye, O. B., & Ekabua, O. O. (2022). *Conceptualisation of Cyberattack Prediction with Deep Learning.*
- Ben Fredj, O., Mihoub, A., Krichen, M., Cheikhrouhou, O., & Derhab, A. (2021). CyberSecurity Attack Prediction: A Deep Learning Approach. 13th International Conference on Security of Information and Networks (SIN 2020), Article 5. https://doi.org/10.1145/3433174.3433614 *Cybersecurity Attack Prediction: A Deep Learning Approach.*

## References

1. UNSW-NB15 Dataset: [https://research.unsw.edu.au/projects/unsw-nb15-dataset](https://research.unsw.edu.au/projects/unsw-nb15-dataset)
