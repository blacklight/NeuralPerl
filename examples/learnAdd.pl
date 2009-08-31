##################################################################
#                                                                #
# learnAdd.pl                                                    #
#                                                                #
# This source uses NeuralNet module to create a neural network   #
# that learns from scratch to add two numbers. You just have to  #
# specify some information for the network to be built, start it #
# with some training sets (i.e. some input values with the       #
# associated expected values), turn those training set into an   #
# XML file to train the network with, and then save the          #
# information about the trained network. Then, you can use the   #
# saved network to sum two numbers whenever you want from any    #
# Perl script (or even C++ applications using Neural++ library). #
# Take a look to doAdd.pl script to see how to load that network #
# and use it.                                                    #
#                                                                #
# by BlackLight, 2009                                            #
# released under GNU GPL licence v.3                             #
#                                                                #
##################################################################


#!/usr/bin/perl 

use strict;
use warnings;
use NeuralNet;

=doc
Build the network. You should specify

 input_size -> Number of neurons in the input layer (the number should
		 be the same of the input values you're going to submit to the
		 network each time, so if you're going to build a network to
		 sum two numbers, the network should have two neurons in its
		 input layer.

 hidden_size -> Number of neurons in the hidden layer. Usually, this
 		number may be the same than the number of neurons in the input
		layer.

 output_size -> Number of neurons in the output layer. If you're going
 		to train the network to sum two numbers, very likely you're
		going to get that sum as output value, and so the output layer
		should contain an only neuron.

 learning_rate -> This value establishes "how fast" the network should
 		learn, i.e. how much it should react to changes of the synaptical
		weights. You can get the optimal learning rate just trying some
		different values. Anyway, the value should be >= 0 and <= 1, even
		if it may be wise to keep that value < 0.1, and rise the number
		of epochs (this makes the network more stable in the training
		phase).

 epochs -> This value represents the number of iterations the network
 		should do in the training phase. The more iterations you do, the
		more the network gets accurate. Anyway, above a certain value
		you won't get great enhancements anymore.

You can also specify more parameters:

 threshold -> This specifies a threshold value for the neurons in the
 		network. The threshold of a neuron is quite similar to the
		threshold voltage of a MOS transistor. This means the neurons
		will activate only their activation value is above that
		threshold, and in that case, their actual activation value
		will be computed as actv-threshold.

 activation_function -> This specifies an alternative activation function.
 		By default, if this parameter is not specified, the identity
		function f(x) = x will be used as activation function. Other often
		used activation functions are the arcotangent, the logistic curve
		and the hyperbolic tangent.

 file -> When used, restore a neural network already trained and saved to
 		an XML file.
=cut

my $net = new NeuralNet(
	input_size => 2,
	hidden_size => 2,
	output_size => 1,
	learning_rate => 0.002,
	epochs => 1000
) or die "Error: $!\n";

print "I'm training the network now, this may take a while\n";

=doc
Here you use the setToXML method to build an XML string from some training
sets. A training set is fundamentally represented as a string in this format:
'input1,input2,...,inputn;expected1,expected2,...expectedm'
where input(i) is an input value, and expected(i) is an expected value as
output. So, if I want to say the network that, when the input values are
2 and 3, it should give me '5' as output, I will just write '2,3;5' as
training set.
=cut

my $xml = $net->setToXML(
	'2,3;5',
	'3,4;7',
	'5,4;9'
);

=doc
Here you use the XML string you've just built to train the network.
=cut

$net->train(
	xml => $xml
);

=doc
Here you can save some information about the network you've just trained on
an XML file. Then, you can load that network whenever and wherever you want
just using that file.
=cut

$net->save('netadd.xml');

print "Network saved to netadd.xml\n";

