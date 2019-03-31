# digit_recognition
a multi-class classification problem for recognizing images of handwritten digits.   
# Testing 
```
java DigitClassifier <numHidden> <learnRate> <maxEpoch> <randomSeed><trainFile> <testFile>
```
where trainFile, and testFile are the names of training and testing datasets,respectively. <br/>
numHidden specifies the number of nodes in the hidden layer (excluding the bias node at the hidden layer). <br/>
learnRate and maxEpoch are the learning rate and the number of epochs that the network will be trained, respectively.<br/>
randomSeed is used to seed the random number generator that initializes the weights and shuffles the training set.<br/>

A sample test command is <br/>
```
java DigitClassifier 5 0.01 100 1 train1.txt test1.txt
```
