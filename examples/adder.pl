#!/usr/bin/perl 

use strict;
use warnings;
use NeuralNet;

my $net = new NeuralNet(
	input_size => 2,
	hidden_size => 2,
	output_size => 1,
	learning_rate => 0.002,
	epochs => 1000
) or die "Error: $!\n";

my $xml = $net->train(
	'2,3;5',
	'3,4;7'
);

$net->setInput(2,3);
$net->propagate;
print $net->getOutput."\n";

