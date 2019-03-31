import java.util.*;

/**
 * The main class that handles the entire network
 * Has multiple attributes each with its own use
 */

public class NNImpl {
    private ArrayList<Node> inputNodes; //list of the input layer nodes.
    private ArrayList<Node> hiddenNodes;    //list of the hidden layer nodes
    private ArrayList<Node> outputNodes;    // list of the output layer nodes

    private ArrayList<Instance> trainingSet;    //the training set

    private double learningRate;    // variable to store the learning rate
    private int maxEpoch;   // variable to store the maximum number of epochs
    private Random random;  // random number generator to shuffle the training set

    /**
     * This constructor creates the nodes necessary for the neural network
     * Also connects the nodes of different layers
     * After calling the constructor the last node of both inputNodes and
     * hiddenNodes will be bias nodes.
     */

    NNImpl(ArrayList<Instance> trainingSet, int hiddenNodeCount, Double learningRate, int maxEpoch, Random random, Double[][] hiddenWeights, Double[][] outputWeights) {
        this.trainingSet = trainingSet;
        this.learningRate = learningRate;
        this.maxEpoch = maxEpoch;
        this.random = random;

        //input layer nodes
        inputNodes = new ArrayList<>();
        int inputNodeCount = trainingSet.get(0).attributes.size();
        int outputNodeCount = trainingSet.get(0).classValues.size();
        for (int i = 0; i < inputNodeCount; i++) {
            Node node = new Node(0);
            inputNodes.add(node);
        }

        //bias node from input layer to hidden
        Node biasToHidden = new Node(1);
        inputNodes.add(biasToHidden);

        //hidden layer nodes
        hiddenNodes = new ArrayList<>();
        for (int i = 0; i < hiddenNodeCount; i++) {
            Node node = new Node(2);
            //Connecting hidden layer nodes with input layer nodes
            for (int j = 0; j < inputNodes.size(); j++) {
                NodeWeightPair nwp = new NodeWeightPair(inputNodes.get(j), hiddenWeights[i][j]);
                node.parents.add(nwp);
            }
            hiddenNodes.add(node);
        }

        //bias node from hidden layer to output
        Node biasToOutput = new Node(3);
        hiddenNodes.add(biasToOutput);

        //Output node layer
        outputNodes = new ArrayList<>();
        for (int i = 0; i < outputNodeCount; i++) {
            Node node = new Node(4);
            //Connecting output layer nodes with hidden layer nodes
            for (int j = 0; j < hiddenNodes.size(); j++) {
                NodeWeightPair nwp = new NodeWeightPair(hiddenNodes.get(j), outputWeights[i][j]);
                node.parents.add(nwp);
            }
            outputNodes.add(node);
        }
    }

    /**
     * Get the prediction from the neural network for a single instance
     * Return the idx with highest output values. For example if the outputs
     * of the outputNodes are [0.1, 0.5, 0.2], it should return 1.
     * The parameter is a single instance
     */

    public int predict(Instance instance) {
    	double max = 0.0;
    	int index = 0;
    	loss(instance);
    	for(int i = 0; i < outputNodes.size(); i++) {
    		if(outputNodes.get(i).getOutput() > max) {
    			max = outputNodes.get(i).getOutput();
    			index = i;
    		}
    	}
        return index;
    }

    

    /**
     * Train the neural networks with the given parameters
     * <p>
     * The parameters are stored as attributes of this class
     */

    public void train() {
        // TODO: add code here
    	for (int epoch = 0; epoch < maxEpoch; epoch++) { 
    		Collections.shuffle(trainingSet, random);
    		double tLoss = 0.0;
    		//denominator for the softMax function
	    	for (Instance ins: trainingSet) {
	    		double dsum = 0.0;
	    		// forward propagation to compute the outputs
	    		// iterate over the input layer
	    		for(int i = 0; i < inputNodes.size() - 1; i++) {
	    			inputNodes.get(i).setInput(ins.attributes.get(i));
	    		}
	    		// iterate over the hidden layer
	    		for (int j = 0; j < hiddenNodes.size() - 1; j++) {
	    			hiddenNodes.get(j).calculateOutput();
	    		}
	    		for (int k = 0; k < outputNodes.size(); k++) {
	    			outputNodes.get(k).calculateOutput();
	    			dsum += outputNodes.get(k).getOutput();
	    		}
	    		for (int k = 0; k < outputNodes.size(); k++) {
	    			outputNodes.get(k).calcsum(dsum);
	    		}
	    		
	    		// backward propagation
	    		// update delta for output layer
	    		for (int k = 0; k < outputNodes.size(); k++) {
	    			outputNodes.get(k).calculateDelta(outputNodes, k, ins.classValues.get(k));
	    		}
	    		// update delta for hidden layer
	    		for (int j = 0; j < hiddenNodes.size(); j++) {
	    			//target value doesn't matter here
	    			hiddenNodes.get(j).calculateDelta(outputNodes, j, 0);
	    		}
	    		// update weight for output layer
	    		for (int k = 0; k < outputNodes.size(); k++) {
	    			outputNodes.get(k).updateWeight(learningRate);
	    		}
	    		// update weight for hidden layer
	    		for (int j = 0; j < hiddenNodes.size() - 1; j++) {
	    			hiddenNodes.get(j).updateWeight(learningRate);
	    		}
	    	}
	    	
	    	//sum up the total loss
	    	for (Instance ins: trainingSet)
	    		tLoss += loss(ins);
	    	
	    	System.out.print("Epoch: " + epoch);
	    	System.out.printf(", Loss: %.8e\n", tLoss/ (double)trainingSet.size());
    	}
    }

    /**
     * Calculate the cross entropy loss from the neural network for
     * a single instance.
     * The parameter is a single instance
     */
    private double loss(Instance instance) {
        // TODO: add code here
    	double sum = 0.0;
    	double dsum = 0.0;
    	//initialize the neural networking
    	for (int i = 0; i < inputNodes.size() - 1; i++) {
    		inputNodes.get(i).setInput(instance.attributes.get(i));
    	}
		// iterate over the hidden layer
		for (int j = 0; j < hiddenNodes.size() - 1; j++) {
			hiddenNodes.get(j).calculateOutput();
		}
		// iterate over the output layer
		for (int k = 0; k < outputNodes.size(); k++) {
			outputNodes.get(k).calculateOutput();
			dsum += outputNodes.get(k).getOutput();
		}
		
		// update the correct output value 
		for (int k = 0; k < outputNodes.size(); k++) {
			outputNodes.get(k).calcsum(dsum);
		}
		//calculate cros-entropy loss
		for (int i = 0; i < outputNodes.size(); i++) {
			sum += (double)instance.classValues.get(i) * (double)Math.log(outputNodes.get(i).getOutput());
		}
    	return -sum;
    }
}

