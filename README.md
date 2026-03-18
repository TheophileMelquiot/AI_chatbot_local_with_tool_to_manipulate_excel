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


**Performance :**

The cross-validation gives a mean accuracy of **0.9248 ± 0.0242** across 5 folds (0.927, 0.941, 0.937, 0.941, 0.877), the spread is reasonable, fold 5 is slightly lower but nothing alarming, the model is consistent. Top-3 accuracy is **0.984** which mean that when the model is wrong, the right label is almost always in its top 3 candidates, so the errors are soft mistakes near the decision boundary rather than completely wrong predictions.

Looking at the classification report, **montant** (f1=0.97), **quantite** (0.97) and **nom_client** (0.92) are the strongest classes, they have large support and very distinctive patterns. **description** is the clear weak point with precision 0.71, recall 0.68 and f1 0.70 : the confusion matrix shows it leaking 0.14 into nom_client and 0.18 into rapport, which makes sense because description columns often contain free-text that look like a client name or a report comment. The **categorie** class at 0.82 recall also carries some confusion but with decent support (39 samples) and a 0.83 f1 it is manageable.

The ROC curves tell a very strong story : **macro OVR AUC = 0.9990**, almost every class hits AUC=1.00. The two lowest are rapport (0.96) and description (0.97) which is consistent with the classification report. The discriminative power of the model is excellent even if the decision boundary on a couple of classes is not perfectly tight.

**Feature Importance :**

The block permutation importance shows that **header** is responsible for the majority of the signal (drop of **0.2285** when permuted), which means the model is mostly learning from the semantic meaning of the column name. The combined (0.0420) and context (0.0420) blocks contribute equally and meaningfully, confirming that encoding header + values together and taking into account neighboring columns does add information beyond the header alone. The **values block alone** (0.0166) is surprisingly weak, meaning the raw value samples without the header context do not carry much discriminative power on their own. The **stats block** (0.0000) shows zero drop when permuted, the 9 statistical features (mean, std, unique ratio ...) do not contribute independently once the embeddings are there, they are likely redundant with what the embeddings already capture.

This is a useful insight for the next iteration : stats could be removed or replaced with more targeted hand-crafted features, and putting more effort into the values embedding or value-level patterns could lift description performance in particular.

**Sanity Tests :**

The label shuffle test gives an accuracy of **0.185**, well below the 0.3 threshold coded in the script. The model trained on randomly shuffled labels learns nothing useful, which confirm there is no data leakage, no hidden shortcut between train and test, the pipeline is clean.

The random header test drops accuracy to **0.471** compared to 0.924 at normal operation, confirming what the feature importance already showed : the header name is the dominant feature and replacing it with noise destroys roughly half the model capacity. The header ablation test (empty string instead of column name) gives **0.499**, very close to the random header result, meaning an empty header and a random garbage header are equivalent for the model, both remove the primary source of signal.

The fact that the model still reaches ~0.49 without any header at all shows the values embedding, combined and context blocks provide a solid secondary signal and the model is not purely relying on one feature. 49% accuracy on an 11-class problem (random chance ≈ 9%, majority class ~30%) is actually meaningful residual performance from the value side alone.




## Conclusion

While the model demonstrates remarkable capabilities, it is crucial to acknowledge its inherent limitations. The neural architecture, while sophisticated, suffers from biases rooted in training data and lacks the ability to understand context in a human-like manner. Furthermore, over-reliance on statistical inferences can lead to errors in judgment and a lack of common sense reasoning. These architectural flaws highlight the need for continuous improvement and critical evaluation of AI models in practical applications.
