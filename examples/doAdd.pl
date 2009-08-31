##################################################################
#                                                                #
# doAdd.pl                                                       #
#                                                                #
# This source uses NeuralNet module to restore a previously      #
# saved neural network. In this case, the script will load a     #
# network to sum two numbers from a previously generated XML     #
# file (take a look to learnAdd.pl to see how it was generated.  #
# Then, it will ask the user two numbers, and will try to        #
# compute their sum using the loaded network.                    #
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
Load the network from the XML file 'netadd.xml'
=cut

my $net = new NeuralNet(
	file => 'netadd.xml'
);

print "Insert a number: ";
my $a = <STDIN>;

print "Insert another number, sir: ";
my $b = <STDIN>;

=doc
Set those values as input values for the neural network.
=cut

$net->setInput($a,$b);

=doc
Propagate the values and get the output value.
=cut

$net->propagate;

=doc
Print the output value using 'output' method. If the network
returned more than one output value, you will use the 'outputs'
method, that returns an array containing the output values.
=cut

print "Ok, more or less the result is ".$net->output."\n";

