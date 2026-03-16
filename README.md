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

That is my final model architecture for this project, i got 9 features to determine the header or not, i use the mini-llmV4-6 which about 90 Mb to embedded the input word, i Scale all my features to be sure that the weight of each features is determine by it's importance and not his scale and because an linear regression model doesn't do it internally, i use an SMOTE to artifically develop the number of sample for the underrepresented class and then i do my first logistic regression on an K-fold of 5 fold to test the accuracy on each partition of the data and then i train the model on my whole DATASET.

<p align="center">
  <img src="images/final_matrix.png" width="45%" />
  <img src="images/final_ROC.png" width="45%" />
</p>








