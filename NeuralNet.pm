#!/usr/bin/perl

#===============================================================================
#
#         FILE:  NeuralNet.pm
#
#  DESCRIPTION:  Main file for Neural package
#
#        FILES:  ---
#         BUGS:  ---
#        NOTES:  ---
#       AUTHOR:  BlackLight (http://0x00.ath.cx), blacklight@0x00.ath.cx
#      COMPANY:  none
#      VERSION:  0.01b
#      CREATED:  18/08/2009 17:29:31
#     REVISION:  ---
#===============================================================================

package NeuralNet;

use strict;
use warnings;
use XML::Parser;

sub actv_f  {
	my $x = shift;
	return $x;
}

sub deriv  {
	my ($f,$x) = @_;
	my $h = 0.0000001;
	return ( (&$f($x+$h) - &$f($x)) / $h );
}

sub momentum  {
	my ($N, $x) = @_;
	return (0.8*$N)/(20*$x + $N);
}

my $actv = \&actv_f;
my $ref_epochs = 0;
my $epochs = 0;

my $input_size  = 0;
my $hidden_size = 0;
my $output_size = 0;

my @input_neurons = ();
my @hidden_neurons = ();
my @output_neurons = ();
my @in_hid_synapses = ();
my @hid_out_synapses = ();
my @expect = ();

sub new  {
	my ($class, %arg) = @_;
	
	$input_size = delete $arg{'input_size'};
	$hidden_size = delete $arg{'hidden_size'};
	$output_size = delete $arg{'output_size'};

	my $l_rate = delete $arg{'learning_rate'};
	my $threshold = delete $arg{'threshold'};

	$epochs = delete $arg{'epochs'};
	$ref_epochs = $epochs;

	Carp::croak ("You must specify the size of input, hidden and output layers ".
			"and at least the learning rate and the number of learning epochs ".
			"for this network") unless ($input_size && $hidden_size && $output_size
				&& $l_rate && $epochs);

	$threshold = 0 unless $threshold;
	$actv = delete $arg{'activation_function'} if defined($arg{'activation_function'});

	for my $i (0..$input_size-1)  {
		my $prop = rand;

		$input_neurons[$i] = {
			propagation_value => $prop,
			activation_value => &$actv($prop),
			threshold => $threshold
		};
	}

	for my $i (0..$hidden_size-1)  {
		my $prop = rand;

		$hidden_neurons[$i] = {
			propagation_value => $prop,
			activation_value => &$actv($prop),
			threshold => $threshold
		};
	}

	for my $i (0..$output_size-1)  {
		my $prop = rand;

		$output_neurons[$i] = {
			propagation_value => $prop,
			activation_value => &$actv($prop),
			threshold => $threshold
		};
	}

	for my $i (0..$input_size-1)  {
		for my $j (0..$hidden_size-1)  {
			$in_hid_synapses[$i][$j] = {
				neuron_in => \$input_neurons[$i],
				neuron_out => \$hidden_neurons[$j],
				delta => 0,
				prev_delta => 0,
				weight => rand
			};
		}
	}

	for my $i (0..$hidden_size-1)  {
		for my $j (0..$output_size-1)  {
			$hid_out_synapses[$i][$j] = {
				neuron_in => \$hidden_neurons[$i],
				neuron_out => \$output_neurons[$j],
				delta => 0,
				prev_delta => 0,
				weight => rand
			};
		}
	}

	my $this = {
		epochs => $epochs,
		threshold => $threshold,
		learning_rate => $l_rate,
		activation_funcion => \&actv,
	};

	bless $this;
	return $this;
}

sub setInput  {
	my ($this, @input) = @_;

	Carp::croak ("Invalid number of input values, different from the number of input neurons")
		unless length(@input) eq length(@input_neurons);

	for my $i (0..length(@input))  {
		$input_neurons[$i]->{propagation_value} = $input[$i];
		$input_neurons[$i]->{activation_value} = &$actv($input[$i]);
	}
}

sub setExpected  {
	my ($this, @e) = @_;
	
	Carp::croak ("Invalid number of expected values, different from the number of output neurons")
		unless length(@e) eq length(@output_neurons);

	@expect = @e;
}

sub propagate  {
	my $this = shift;

	for my $neuron (@hidden_neurons)  {
		$neuron->{propagation_value} = -($this->{threshold});
	}

	for my $neuron (@output_neurons)  {
		$neuron->{propagation_value} = -($this->{threshold});
	}

	for my $i (0..$input_size-1)  {
		for my $j (0..$hidden_size-1)  {
			${$in_hid_synapses[$i][$j]->{neuron_out}}->{propagation_value} +=
				$in_hid_synapses[$i][$j]->{weight} * ${$in_hid_synapses[$i][$j]->{neuron_in}}->{activation_value};
		}
	}

	for my $neuron (@hidden_neurons)  {
		$neuron->{activation_value} = &$actv($neuron->{propagation_value});
	}

	for my $i (0..$hidden_size-1)  {
		for my $j (0..$output_size-1)  {
			${$hid_out_synapses[$i][$j]->{neuron_out}}->{propagation_value} +=
				$hid_out_synapses[$i][$j]->{weight} * ${$hid_out_synapses[$i][$j]->{neuron_in}}->{activation_value};
		}
	}
	
	for my $neuron (@output_neurons)  {
		$neuron->{activation_value} = &$actv($neuron->{propagation_value});
	}
}

sub error  {
	my $err = 0;
	my $i = 0;

	for my $neuron (@output_neurons)  {
		my $out = $neuron->{activation_value};
		my $ex  = $expect[$i++];
		$err += ($out-$ex)*($out-$ex);
	}

	$err /= 2;
	return $err;
}

sub updateWeights  {
	my $this = shift;
	my $Dk = 0;

	for my $i (0..$output_size-1)  {
		my $neuron = $output_neurons[$i];
		my $out_delta = 0;
		my $z = $neuron->{activation_value};
		my $d = $expect[$i];
		my $f = deriv($actv, $neuron->{propagation_value});

		for my $j (0..$hidden_size-1)  {
			my $y = ${$hid_out_synapses[$j][$i]->{neuron_in}}->{activation_value};
			my $beta = momentum($ref_epochs, $ref_epochs - $epochs);

			if ($ref_epochs - $epochs > 0)  {
				$out_delta = (- $this->{learning_rate}) * ($z-$d) * $f * $y +
					$beta * $hid_out_synapses[$j][$i]->{prev_delta};
			} else {
				$out_delta = (- $this->{learning_rate}) * ($z-$d) * $f * $y;
			}

			Carp::croak ("Invalid synaptical variation > 1 while updating synaptical weights")
				if ($out_delta > 1);

			$Dk += ( ($z-$d) * $f * $hid_out_synapses[$j][$i]->{weight} );
			$hid_out_synapses[$j][$i]->{prev_delta} = $hid_out_synapses[$j][$i]->{delta};
			$hid_out_synapses[$j][$i]->{delta} = $out_delta;
		}
	}

	for my $i (0..$hidden_size-1)  {
		my $neuron = $hidden_neurons[$i];
		my $hidden_delta = 0;
		my $d = deriv($actv, $neuron->{propagation_value}) * $Dk;

		for my $j (0..$input_size-1)  {
			my $x = ${$in_hid_synapses[$j][$i]->{neuron_in}}->{activation_value};
			my $beta = momentum($ref_epochs, $ref_epochs - $epochs);

			if ($ref_epochs - $epochs > 0)  {
				$hidden_delta = (- $this->{learning_rate}) * $d * $x +
					$beta * $in_hid_synapses[$j][$i]->{prev_delta};
			} else {
				$hidden_delta = (- $this->{learning_rate}) * $d * $x;
			}

			Carp::croak ("Invalid synaptical variation > 1 while updating synaptical weights")
				if ($hidden_delta > 1);

			$in_hid_synapses[$j][$i]->{prev_delta} = $in_hid_synapses[$j][$i]->{delta};
			$in_hid_synapses[$j][$i]->{delta} = $hidden_delta;
		}
	}

	for my $i (0..$input_size-1)  {
		for my $j (0..$hidden_size-1)  {
			$in_hid_synapses[$i][$j]->{weight} += $in_hid_synapses[$i][$j]->{delta};
			$in_hid_synapses[$i][$j]->{delta} = 0;
		}
	}

	for my $i (0..$hidden_size-1)  {
		for my $j (0..$output_size-1)  {
			$hid_out_synapses[$i][$j]->{weight} += $hid_out_synapses[$i][$j]->{delta};
			$hid_out_synapses[$i][$j]->{delta} = 0;
		}
	}
}

my $in_network  = 0;
my $in_training = 0;
my $in_input  = 0;
my $in_output = 0;
my @xml_input  = ();
my @xml_expect = ();
my $num_sets  = 0;
my $in_index  = 0;
my $out_index = 0;

sub tagstart  {
	my ($parser, $name, %attr) = @_;

	for ($name)  {
		if (/^network$/)  {
			Carp::croak ("Invalid \"network\" tag found inside another network tag")
				if $in_network ne 0;

			$in_network = 1;
		} elsif (/^training$/)  {
			Carp::croak ("Invalid \"training\" tag found inside another training tag")
				if $in_training ne 0;

			$in_training = 1;
		} elsif (/^input$/)  {
			Carp::croak ("Invalid \"input\" tag found inside another input tag")
				if $in_input ne 0;

			$in_input = 1;
		} elsif (/^output$/)  {
			Carp::croak ("Invalid \"output\" tag found inside another output tag")
				if $in_output ne 0;

			$in_output = 1;
		}
	}
}

sub tagparse  {
	my ($parser, $data) = @_;

	$xml_input [$num_sets][$in_index ++] = $data if $in_input;
	$xml_expect[$num_sets][$out_index++] = $data if $in_output;
}

sub tagend  {
	my ($parser, $name) = @_;

	$in_network = 0 if $name =~ /^network$/;
	$in_input   = 0 if $name =~ /^input$/;
	$in_output  = 0 if $name =~ /^output$/;

	if ($name =~ /^training$/)  {
		$in_training = 0;
		$in_index  = 0;
		$out_index = 0;
		$num_sets++;
	}
}

sub trainFromXML  {
	my ($this, $xml) = @_;
	$xml =~ tr/[A-Z]/[a-z]/;

	my $parser = new XML::Parser;
	$parser->setHandlers(
		Start => \&tagstart,
		Char  => \&tagparse,
		End   => \&tagend
	);

	$parser->parse($xml);
	my @in = ();
	my @ex = ();

	for my $i (0..@xml_input-1)  {
		for my $j (0..$input_size-1)  {
			push @in, $xml_input[$i][$j];
		}

		for my $j (0..$output_size-1)  {
			push @ex, $xml_expect[$i][$j];
		}

		$this->setInput(@in);
		$this->setExpected(@ex);

		while ($epochs--)  {
			$this->propagate;
			$this->updateWeights;
		}

		$epochs = $this->{epochs};
		@in = ();
		@ex = ();
	}
}

sub train  {
	my ($this, %trainer) = @_;

	if (defined $trainer{training_set})  {
		my @wut = split ';', $trainer{training_set};
		$this->setInput(split ',', $wut[0]);
		$this->setExpected(split ',', $wut[1]);

		while ($epochs--)  {
			$this->propagate;
			$this->updateWeights;
		}

		$epochs = $this->{epochs};
	} elsif (defined $trainer{xml})  {
		$this->trainFromXML($trainer{xml});
	} elsif (defined $trainer{xmlfile})  {
		my $xml = '';
		open IN, "< ".$trainer{xmlfile} or Carp::croak("Unable to open ".$trainer{xmlfile}.": $!\n");
		$xml .= $_ for (<IN>);
		close IN;

		$this->trainFromXML($xml);
	}
}

sub getOutput  {
	return $output_neurons[0]->{activation_value};
}

sub getOutputs  {
	my @out;

	for my $neuron (@output_neurons)  {
		push @out, $neuron->{activation_value};
	}

	return @out;
}

sub setToXML  {
	my $xml = '';
	$xml .=
		"<?xml version=\"1.0\" encoding=\"iso-8859-1\"?>\n".
		"<!DOCTYPE NETWORK SYSTEM \"http://blacklight.gotdns.org/prog/neuralpp/trainer.dtd\">\n".
		"<!-- Automatically generated by Neural++ library - by BlackLight -->\n\n".
		"<NETWORK>\n";
}

1;

