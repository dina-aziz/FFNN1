#include<iostream>
#include<cmath>
#include<iomanip>
#include<ctime>
#include<conio.h>

#define ARRAY_SIZE(array) (sizeof((array))/sizeof((array[0])))    // gets the length of an array
using namespace std;
float randomizeAround(float n)  // returns a random number around the argument n (the number may be multiplied by -1)
{
	int sign=rand()%2;
	int addSub=rand()%2;
	int r=rand()%30;
	float range=float(r);
	range/=100;
	if(addSub==0) n+=range;
	else n-=range;
	if(sign==0) return n;
	else return -1*n;
}
class logisticNeuron    // this class represents a neuron
{
private:
	float * weights; 
	float output;
public:
	logisticNeuron()
	{
		//cout<<"\n creating neuron..";
	}
	logisticNeuron(int numOfLinks,float initialWeight)
	{
		weights=new float[numOfLinks+1];
		for(int j=0;j<numOfLinks+1;j++) weights[j]=randomizeAround(initialWeight); // initialize every weight with a number around: initialWeight
		//cout<<"\n creating neuron..";
	}
	void activate(float input[],int size) // takes an array of inputs given to a neuron and returns the result of the activation function
	{
		float net=weights[size];
		for(int j=0;j<size;j++) net+=input[j]*weights[j];
        output=(2/(1+pow(2.718f,-1*net)))-1;                 // bipolar
		//output=1/(1+pow(2.718f,-1*net));                //unipolar
	}
	float getOutput(){return output;}
	void setWeight(int n,float w){weights[n]=w;}
	float getWeight(int n){return weights[n];}
	void setNeuronData(int numOfLinks,float initialWeight)   // used to specify the number of links to a neuron, and set the weights of these links
	{
		weights=new float[numOfLinks+1];
		for(int j=0;j<numOfLinks+1;j++) weights[j]=randomizeAround(initialWeight);
		//cout<<"\n setting neuron data..";
	}
};
class ANN
{
private:
	logisticNeuron ** network;     // ntework is composed of a 2D array of neurons
	float *output;                 //vector representing the outputs of the output layer
	float numOfInputs;
    int depth;        // number of layers in the network
	int *Layers;      // a vector representing the number of neurons in each layer
	float **Outputs;              // 2D array of outputs of each neuron 
	float **dE_dZ;                // 2D array represents dE/dZ at every neuron
	float **dE_dY;                // 2D array represents dE/dY at every neuron
	float **dE_dW;                // 2D array represents dE/dW to every weight, at every layer, the number of weights = number of neurons at this layer * the number of neurons at the layer below
	float **dE_dW_bias;           // 2D array represents the dE/dW of the bias weights : i have treated these weights separately from other weights 
public:
	ANN(int layers[],int Depth,float initialWeight) // layers : number of neurons at each layer
	{
		output= new float[layers[Depth-1]];
		numOfInputs=layers[0];  // number of inputs is equal to the number of netrons in the input layer
		depth=Depth;
		Layers=new int[depth];    // keeping data about the dimensions of the network
		for(int j=0;j<depth;j++) Layers[j]=layers[j];
		network =new logisticNeuron* [depth];   // filling the network with  neurons
		for(int j=0;j<depth;j++)
			network[j]= new logisticNeuron [layers[j]];
		for(int j=0;j<depth;j++)            // setting data for neurons in the network
			for(int i=0;i<Layers[j];i++)
			{
				if(j==0)network[j][i].setNeuronData(layers[0],initialWeight);  // layers[0]: input layer , layers[depth-1]: output layer 
				else network[j][i].setNeuronData(layers[j-1],initialWeight);
			}
		Outputs = new float* [depth];    // defining arrays that dE/dW (and those for bias weights) , dE/dY , dE/dZ with the needed dimensions
		for(int k=0;k<depth;k++) Outputs[k]=new float [layers[k]];
		dE_dZ = new float*[depth];
		for(int k=0;k<depth;k++) dE_dZ[k]=new float [layers[k]];
		dE_dY = new float*[depth];
		for(int k=0;k<depth;k++) dE_dY[k]=new float [layers[k]];
		dE_dW = new float*[depth];
		for(int k=depth-1;k>0;k--) dE_dW[k]=new float [layers[k]*layers[k-1]];
		dE_dW[0]=new float [layers[0]*layers[0]];
		dE_dW_bias = new float*[depth];
		for(int k=0;k<depth;k++) dE_dW_bias[k]=new float [layers[k]];

	}
	float* getOutput(){return output;}   // returns the array of outputs of the output layer
	void propagate(float input[],int inputSize)  // foreward propagation through the network , the array of inputs is given
	{
		//if(numOfInputs!=inputSize) {cout<<"\nerror, number of inputs doesn't match."<<endl;return;}
		//else 
		//{
			for(int j=0;j<depth;j++)
			{
				for(int i=0;i<Layers[j];i++)
				{
					if(j==0)network[j][i].activate(input,inputSize);  // activate with inputs
					else network[j][i].activate(Outputs[j-1],Layers[j-1]); // activate with the outputs of the layer below
					Outputs[j][i]=network[j][i].getOutput();  // save the neuron output in the 2D output array
				}
			}output=Outputs[depth-1];  // the outputs of the output layer 
		//}
	}
	void backPropagate(float target[],int tsize,float input[],int isize,float lr=1)  // backpropagation through the network
	{  
		if((tsize!=Layers[depth-1]) && (isize!=Layers[0])){cout<<"the target or input size doesn't match";return;}
		static float error=0;     // computing the error produced by the foreward propagation of the inputs
		float e=0;
		for(int j=0;j<Layers[depth-1];j++) e+=pow((target[j]-output[j]),2);
		e*=0.5;
		error+=e;
		//cout<<"\nError: "<<error;
		for(int j=depth-1;j>=0;j--)
		{
			int n=0;
			for(int i=0;i<Layers[j];i++)
			{
				if(j==depth-1)   // setting dE/dY and dE/dZ for the output layer
				{
					dE_dZ[j][i]=0.5*(target[i]-Outputs[j][i])*(1-Outputs[j][i]*Outputs[j][i]);
				}
				else    // setting dE/dY and dE/dZ for the layers below
				{
					dE_dY[j][i]=0;
					for(int k=0;k<Layers[j+1];k++)
						dE_dY[j][i]+=dE_dZ[j+1][k]*(network[j+1][k].getWeight(i)); 
					dE_dZ[j][i]=0.5*(1-Outputs[j][i]*Outputs[j][i])*dE_dY[j][i];
				}
				
				if(j<depth-1)   // setting dE/dW for the all layers except the input layer 
				{
					for(int h=n;h<Layers[j+1]+n;h++)
					{	
						dE_dW[j+1][h]+=Outputs[j][i]*dE_dZ[j+1][h-n]*lr; // += for batch learning
					}
				}
				n+=Layers[j+1];
			}
			if(j<depth-1)
			{
				for(int m=0;m<Layers[j+1];m++)dE_dW_bias[j+1][m]+=1*dE_dZ[j+1][m]*lr;   // setting dE/dW of bias weights for the all layers except the input layer 
			}
		}
		for(int h=0;h<Layers[0]*Layers[0];h++) // setting dE/dW for the input layer 
			dE_dW[0][h]+=input[h/Layers[0]]*dE_dZ[0][h/Layers[0]]*lr;
		for(int m=0;m<Layers[0];m++)dE_dW_bias[0][m]+=1*dE_dZ[0][m]*lr;   // setting dE/dW of bias weights for the input layer 
	}  
	void update(float e=1)  // using dE/dW and dE/dW(of the bias weights) to update all the weights
	{
		float newWeight=0.0;
		for(int j=depth-1;j>=0;j--)
		{
			int k=0;
			for(int i=0;i<Layers[j];i++)
			{    
				if(j>0)   // setting new weights in all layers except the input layer
					for(int h=0;h<Layers[j-1];h++)
					{
						newWeight=network[j][i].getWeight(h);
						newWeight+=(e*dE_dW[j][k]);
						//newWeight+=dE_dW[j][k];
						network[j][i].setWeight(h,newWeight);
						dE_dW[j][k]=0.0;
						k++;
					}
				else   // setting new weights in the input layer
					for(int h=0;h<Layers[0];h++)
					{
						newWeight=network[j][i].getWeight(h);
						newWeight+=(e*dE_dW[j][k]);
						//newWeight+=dE_dW[j][k];
						network[j][i].setWeight(h,newWeight);
						dE_dW[j][k]=0.0;
						k++;
					}
			}
			if(j>0)  // setting new bias weights in all layers except the input layer
			{
				for(int m=0;m<Layers[j];m++)
				{
					newWeight=network[j][m].getWeight(Layers[j-1]);
					newWeight+=(e*dE_dW_bias[j][m]);
					//newWeight+=dE_dW_bias[j][m];
					network[j][m].setWeight(Layers[j-1],newWeight);
					dE_dW_bias[j][m]=0.0;
				}
			}
			else   // setting new bias weights in the input layer
			{
				for(int m=0;m<Layers[0];m++)
				{
					newWeight=network[j][m].getWeight(Layers[0]);
					newWeight+=(e*dE_dW_bias[j][m]);
					//newWeight+=dE_dW_bias[j][m];
					network[j][m].setWeight(Layers[j],newWeight);
					dE_dW_bias[j][m]=0.0;
				}
			}
		}
	}
	void showWeightErrorDerivatives()   // display dE/dW
	{
		cout<<endl;
		for(int j=depth-1;j>0;j--)
		{
			if(j>0)
				for(int i=0;i<Layers[j]*Layers[j-1];i++)
					cout<<setiosflags(ios::fixed)<<setprecision(4)<<dE_dW[j][i]<<"   "<<endl;
			else
				for(int i=0;i<Layers[j]*Layers[j];i++)
					cout<<setiosflags(ios::fixed)<<setprecision(4)<<dE_dW[j][i]<<"   "<<endl;
			for(int i=0;i<Layers[j];i++)
					cout<<setiosflags(ios::fixed)<<setprecision(4)<<dE_dW_bias[j][i]<<"   "<<endl;
			cout<<endl;
		}
		for(int i=0;i<Layers[0]*2;i++) cout<<dE_dW[0][i]<<"  "<<endl;
		cout<<endl;
	}

	void showOutputErrorDerivatives()  // display dE/dY
	{
		for(int j=depth;j>=0;j--)
		{
			for(int i=0;i<Layers[j];i++)
				cout<<setiosflags(ios::fixed)<<setprecision(4)<<dE_dY[j][i]<<"   "<<endl;
			cout<<endl;
		}
	}
	void showNetErrorDerivatives()  // display dE/dZ
	{
		for(int j=depth;j>=0;j--)
		{
			for(int i=0;i<Layers[j];i++)
				cout<<setiosflags(ios::fixed)<<setprecision(4)<<dE_dZ[j][i]<<"   "<<endl;
			cout<<endl;
		}
	}
	void showWeights()  // display weights
	{
		cout<<endl;
		float newWeight=0.0;
		for(int j=depth-1;j>=0;j--)
		{
			int k=0;
			for(int i=0;i<Layers[j];i++)
			{
				if(j>0)
					for(int h=0;h<Layers[j-1];h++)
					{
						newWeight=network[j][i].getWeight(h);
						cout<<setiosflags(ios::fixed)<<setprecision(4)<<newWeight;
						cout<<endl;
					}
				else
					for(int h=0;h<Layers[0];h++)
					{
						newWeight=network[j][i].getWeight(h);
						cout<<setiosflags(ios::fixed)<<setprecision(4)<<newWeight;
						cout<<endl;
					}
			}
			if(j>0)
				for (int m=0;m<Layers[j];m++)
				{
					newWeight=network[j][m].getWeight(Layers[j-1]);
					cout<<setiosflags(ios::fixed)<<setprecision(4)<<newWeight;
					cout<<endl;
				}
			else
				for (int m=0;m<Layers[j];m++)
				{
					newWeight=network[j][m].getWeight(Layers[0]);
					cout<<setiosflags(ios::fixed)<<setprecision(4)<<newWeight;
					cout<<endl;
				}
		}
	}
};



int main()
{
	srand(time(NULL));
	float X[2];  // discribing the input dimensions 
	int net[]={4,1};  // discribing the network dimensions  
	ANN ann1(net,ARRAY_SIZE(net),0.5);  
	float* output;
	float T[1];
	float lr=0.001;
	for(int p=0;p<100000;p++)
	{
	for(int t=0;t<4;t++)
	{
		if(t==0){X[0]=-1;X[1]=-1;}
		if(t==0){X[0]=-1;X[1]=1;}
		if(t==0){X[0]=1;X[1]=-1;}
		if(t==0){X[0]=1;X[1]=1;}
		//cout<<endl<<t<<":       x0: "<<X[0]<<"      x1: "<<X[1];
		ann1.propagate(X,ARRAY_SIZE(X));
		output=ann1.getOutput();
		//for(int j=0;j<net[ARRAY_SIZE(net)-1];j++)
			//cout<<"     Output: "<<output[j];
		if(X[0] != X[1])T[0]=1.0;   // specifying the target 
		else T[0]=-1.0;
		//cout<<"         Target: "<<T[0];
		//cout<<endl<<"-------------------------------------------------------------------------";
		ann1.backPropagate(T,ARRAY_SIZE(T),X,ARRAY_SIZE(X),lr);
		//cout<<endl;
		//cout<<" \nWeights: ";
		//cout<<endl<<endl<<"-------------------------------------------------------------------------";
	}
	ann1.update();
	}
	X[0]=1;
	X[1]=1;
	ann1.propagate(X,ARRAY_SIZE(X));
	cout<<endl<<" x0: "<<X[0]<<"      x1: "<<X[1];
	for(int j=0;j<net[ARRAY_SIZE(net)-1];j++)
		cout<<"          Output: "<<output[j]<<endl;
	X[0]=1;
	X[1]=-1;
	ann1.propagate(X,ARRAY_SIZE(X));
	cout<<endl<<" x0: "<<X[0]<<"      x1: "<<X[1];
	for(int j=0;j<net[ARRAY_SIZE(net)-1];j++)
		cout<<"          Output: "<<output[j]<<endl;
	X[0]=-1;
	X[1]=1;
	ann1.propagate(X,ARRAY_SIZE(X));
	cout<<endl<<" x0: "<<X[0]<<"      x1: "<<X[1];
	for(int j=0;j<net[ARRAY_SIZE(net)-1];j++)
		cout<<"          Output: "<<output[j]<<endl;
	X[0]=-1;
	X[1]=-1;
	ann1.propagate(X,ARRAY_SIZE(X));
	cout<<endl<<" x0: "<<X[0]<<"      x1: "<<X[1];
	for(int j=0;j<net[ARRAY_SIZE(net)-1];j++)
		cout<<"          Output: "<<output[j]<<endl;
	//getch();
	return 0;
}


