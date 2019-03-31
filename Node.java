import java.util.*;

/**
 * Class for internal organization of a Neural Network.
 * There are 5 types of nodes. Check the type attribute of the node for details.
 * Feel free to modify the provided function signatures to fit your own implementation
 */

public class Node {
    private int type = 0; //0=input,1=biasToHidden,2=hidden,3=biasToOutput,4=Output
    public ArrayList<NodeWeightPair> parents = null; //Array List that will contain the parents (including the bias node) with weights if applicable

    private double inputValue = 0.0;
    private double outputValue = 0.0;
    private double outputGradient = 0.0;
    private double delta = 0.0; //input gradient

    //Create a node with a specific type
    Node(int type) {
        if (type > 4 || type < 0) {
            System.out.println("Incorrect value for node type");
            System.exit(1);

        } else {
            this.type = type;
        }

        if (type == 2 || type == 4) {
            parents = new ArrayList<>();
        }
    }

    //For an input node sets the input value which will be the value of a particular attribute
    public void setInput(double inputValue) {
        if (type == 0) {    //If input node
            this.inputValue = inputValue;
        }
    }

    
    /**
     * helper method to update the correct outputValue w/ denominator
     */
    public void calcsum(double sum){
    	outputValue = outputValue / sum;
 
    }
    
    /**
     * Calculate the output of a node.
     * You can get this value by using getOutput()
     */
    public void calculateOutput() {
        if (type == 2 || type == 4) {   //Not an input or bias node
        	double sum = 0.0;
        	for (int i = 0; i < parents.size(); i++)
        		//w_i * a_i
        		sum += (parents.get(i).weight * parents.get(i).node.getOutput());
        	if(type == 2) 
        		//using reLu function
        		outputValue = Math.max(0.0, sum);
        	else
        		//numerator of the softMax function
        		outputValue = Math.exp(sum);
        }
    }
    

    //Gets the output value
    public double getOutput() {

        if (type == 0) {    //Input node
            return inputValue;
        } else if (type == 1 || type == 3) {    //Bias node
            return 1.00;
        } else {
            return outputValue;
        }

    }

    //Calculate the delta value of a node.
    public void calculateDelta(ArrayList<Node> outputNode, int idx, double targetValue) {
        if (type == 2 || type == 4)  {
        	// when in output layer
        	if (type == 4)
        		delta = targetValue - outputValue;
        	else {
        		// when in hidden layer
        		double sum = 0.0;
        		//for each output node
        		for (int i = 0; i < outputNode.size(); i++) 
        			//delta_K * weight_jk
        			sum += outputNode.get(i).parents.get(idx).weight * outputNode.get(i).delta;
        		//the derivative of reLu function
        		if (outputValue <= 0)
        			delta = 0;
        		else 
        			delta = sum;
        	}
        		
        }
    }


    //Update the weights between parents node and current node
    public void updateWeight(double learningRate) {
        if (type == 2 || type == 4) {
        	for (int i = 0; i < parents.size(); i++) 
        		//update weight in each connection
        		parents.get(i).weight += learningRate * parents.get(i).node.getOutput() * delta;
        
        }
    }
}


