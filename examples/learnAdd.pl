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

my $xml = $net->setToXML(
	'2,3;5',
	'3,4;7',
	'5,4;9'
);

$net->train(
	xml => $xml
);

$net->save('netadd.xml');
print "Network saved to netadd.xml\n";

