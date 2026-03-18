# AI_chatbot_local_with_tool_to_manipulate_excel

The goal is in the end to make an simple chatbot with tool to manipulate, extract information from an excel file or multiple excel file to et inforamtion from them or to transform this data into an new file , much quicker than doing everything by end.

Timeline of the project :

                                    Making an extract model in python to capture the wanted data 
                                                                  |
                                Making the interface and an chatbot to pass input into the extracting model 
                                                                  |
                                                          Deploy the model 
                                                                  |
                                                      Adding new tools gradually 


Not putting all the detail step for each of this projet like fine tuning, dataset generation ...







#

# First step : Extract model

<p align="center">
  <img src="images/Figure_1.png" width="45%" />
  <img src="images/Figure_roc.png" width="45%" />
</p>

The model present already some pretty good result that come from his architecture because I'm using an SentenceTransformer which encode with good result the word. Thus my model is really doing embedding + logistic regression, its already a linearly classifier on embedding. The pipeline looking like that : 

                                                        header embedding
                                                              +
                                                        value embedding
                                                              +
                                                        column statistics
                                                              =
                                                      logistic regression
                                                        
But it encounter some basic error such has too much representation of certain class and some other under represented which make noise in the dataset. The training set and testing is not really independant, not k cross validation, normalising my stats and inputting and confidence threshold ... 


# Infusing new features to the model, SMOTE, K-fold validations, Normalisation of all the features and multiple sanity test to check for data leakage or overfitting

Full pipeline of how i train my model : 

                                                       Header + sample_values
                                                              ↓
                                                         Feature engineering
                                                          ├ embeddings
                                                          ├ stats
                                                          ├ patterns
                                                          └ context
                                                              ↓
                                                      Feature vector (1545)
                                                              ↓
                                                        StandardScaler
                                                              ↓
                                                            SMOTE
                                                              ↓
                                                      logistic regression
                                                              ↓
                                                        Cross Validation
                                                              ↓
                                                          evaluation
                                                              =
                                                          Model saved 

header embedding 384 + values embedding 384 + combined 384 + context 384 + stats 9 = 1545 features


**Feature Engineering (The Foundation):**

The reason i create 4 different embedding types instead of just one simple embedding is because each capture different aspect of the data. The header embedding understand the semantic meaning of the column name, the value embedding capture what the actual data look like, the combined embedding give context of both together, and the context embedding take into account the surrounding columns. This multi-perspective approach is crucial because a "Price" column with values [100, 200, 300] look very different from a "Price" column with values [0.1, 0.2, 0.3], same semantic meaning but completely different statistical pattern.

The 9 stats features i extract (mean, median, std, min, max, skewness, kurtosis, percentage of nulls, unique value count) serve as a numerical fingerprint of the column. While embeddings capture the semantic meaning, statistics capture the mathematical reality of the data distribution. Together, they give the model both "what the data is supposed to mean" and "how it actually behave".

**StandardScaler (Why Normalize?):**

Before feeding these 1545 features into the logistic regression, i normalize them with StandardScaler. Here's why: if one feature span from 0 to 1,000,000 and another from 0 to 1, the model would naturally weight the larger range feature more heavily during training, not because it's more important but simply because of its scale. By transforming all features to have mean=0 and std=1, i ensure that logistic regression can fairly evaluate which features are truly predictive of a column being a header.

**SMOTE (Handling Imbalance):**

In real-world data, headers are much rarer than non-header rows (maybe 5-10% headers, 90-95% regular data). Training on imbalanced data bias the model toward predicting "not a header" because it get rewarded more often. SMOTE synthetically generate new samples for the minority class (headers) by interpolating between existing minority samples in the feature space. This don't create fake data but rather fill the gap in the feature space so the model learn a better decision boundary.

**K-Fold Cross Validation (Why Not Just Train/Test Split?):**

With a simple train/test split, you risk luck – maybe your test set happen to contain easier or harder examples. K-fold (5 folds in your case) partition the data into 5 subset, train 5 different model each using 4 fold for training and 1 for testing, then average the results. This give you 5 independent measurement of performance and reveal if your model is consistently good or just lucky on one particular split.

**Final Training:**

Once you confirm through cross-validation that your model architecture and preprocessing strategy are sound (checking both the metrics and the ROC curve for overfitting), you train one final model on the entire dataset. This give the model maximum data to learn from, and you already know from cross-validation that this approach should generalize well.



<p align="center">
  <img src="images/final_matrix.png" width="45%" />
  <img src="images/final_ROC.png" width="45%" />
</p>








