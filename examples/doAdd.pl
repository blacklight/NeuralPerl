#!/usr/bin/perl 

use strict;
use warnings;
use NeuralNet;

my $net = new NeuralNet(
	file => 'netadd.xml'
);

$net->setInput(10,20);
$net->propagate;
print $net->getOutput."\n";

