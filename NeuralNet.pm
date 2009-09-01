#!/usr/bin/perl

#===============================================================================
#
#         FILE:  NeuralNet.pm
#
#  DESCRIPTION:  NeuralNet package
#
#        FILES:  ---
#         BUGS:  none known
#        NOTES:  ---
#       AUTHOR:  BlackLight (http://0x00.ath.cx), blacklight@autistici.org
#      VERSION:  0.1
#      CREATED:  18/08/2009 17:29:31
#     REVISION:  ---
#      LICENCE:  GPL v.3
#===============================================================================

package NeuralNet;

$VERSION = '0.1';
sub Version  { $VERSION; }

use strict;
use warnings;
use XML::Parser;

my $actv = \&actv_f;
my $ref_epochs = 0;
my $epochs = 0;

my $input_size  = 0;
my $hidden_size = 0;
my $output_size = 0;
my $threshold = 0;
my $l_rate = 0;

my @input_neurons = ();
my @hidden_neurons = ();
my @output_neurons = ();
my @in_hid_synapses = ();
my @hid_out_synapses = ();
my @expect = ();

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

sub new  {
	my ($class, %arg) = @_;
	
	$input_size  = delete $arg{'input_size'} if defined $arg{'input_size'};
	$hidden_size = delete $arg{'hidden_size'} if defined $arg{'hidden_size'};
	$output_size = delete $arg{'output_size'} if defined $arg{'output_size'};

	$l_rate = delete $arg{'learning_rate'} if defined $arg{'learning_rate'};
	$threshold = delete $arg{'threshold'} if defined $arg{'threshold'};

	$epochs = delete $arg{'epochs'} if defined $arg{'epochs'};
	$ref_epochs = $epochs;
	$actv = delete $arg{'activation_function'} if defined($arg{'activation_function'});

	my $fname = '';
	$fname = delete $arg{'file'} if defined $arg{'file'};

	if ($fname)  {
		my $xml = '';

		open IN, "< $fname"
			or Carp::croak ("Unable to read from $fname");

		for (<IN>)  {
			$_ =~ tr/[A-Z]/[a-z]/ if $_ !~ /^<!DOCTYPE/;
			$xml .= $_;
		}

		close IN;

		my $parser = new XML::Parser;
		$parser->setHandlers(
			Start => \&loadXMLTagStart,
			End => \&loadXMLTagEnd
		);

		$parser->parse($xml);
	} else {
		Carp::croak ("You must specify the size of input, hidden and output layers ".
				"and at least the learning rate and the number of learning epochs ".
				"for this network")
			unless ($input_size && $hidden_size && $output_size && $l_rate && $epochs);

		for my $i (0..$input_size-1)  {
			$input_neurons[$i] = {
				propagation_value => 0,
				activation_value => &$actv(0),
				threshold => $threshold
			};
		}

		for my $i (0..$hidden_size-1)  {
			$hidden_neurons[$i] = {
				propagation_value => 0,
				activation_value => &$actv(0),
				threshold => $threshold
			};
		}

		for my $i (0..$output_size-1)  {
			$output_neurons[$i] = {
				propagation_value => 0,
				activation_value => &$actv(0),
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

sub train  {
	my $this = shift;
	my (%trainer) = @_;

	if (defined $trainer{xml})  {
		$this->trainFromXML($trainer{xml});
		return;
	} elsif (defined $trainer{xmlfile})  {
		my $xml = '';
		open IN, "< ".$trainer{xmlfile} or Carp::croak("Unable to open ".$trainer{xmlfile}.": $!\n");
		$xml .= $_ for (<IN>);
		close IN;

		$this->trainFromXML($xml);
		return;
	}

	my (@set) = @_;

	for (@set)  {
		my @wut = split ';';
		$this->setInput(split ',', $wut[0]);
		$this->setExpected(split ',', $wut[1]);

		while ($epochs--)  {
			$this->propagate;
			$this->updateWeights;
		}

		$epochs = $this->{epochs};
	}
}

sub output  {
	return $output_neurons[0]->{activation_value};
}

sub outputs  {
	my @out;

	for my $neuron (@output_neurons)  {
		push @out, $neuron->{activation_value};
	}

	return @out;
}

sub setToXML  {
	my ($this, @set) = @_;
	my $id  = 0;
	my $xml = '';

	$xml .=
		"<?xml version=\"1.0\" encoding=\"iso-8859-1\"?>\n".
		'<!DOCTYPE NETWORK SYSTEM "http://blacklight.gotdns.org/prog/neuralpp/trainer.dtd">'."\n".
		"<!-- Automatically generated by NeuralPerl module - by BlackLight -->\n\n".
		"<NETWORK>\n";

	for (@set)  {
		$xml .= "\t<TRAINING ID=\"".$id++."\">\n";

		my @wut = split ';';

		for (split ',', $wut[0])  {
			$xml .= "\t\t<INPUT ID=\"".$id++."\">$_</INPUT>\n";
		}

		for (split ',', $wut[1])  {
			$xml .= "\t\t<OUTPUT ID=\"".$id++."\">$_</OUTPUT>\n";
		}

		$xml .= "\t</TRAINING>\n\n";
	}

	$xml .= "</NETWORK>\n";
	return $xml;
}

sub save  {
	my ($this, $file, $name) = @_;
	my $xml = '';

	Carp::croak ("Output file not specified")
		unless $file;

	$name = "NeuralNetwork" unless $name;

	$xml .=
		"<?xml version=\"1.0\" encoding=\"iso-8859-1\"?>\n".
		'<!DOCTYPE NETWORK SYSTEM "http://blacklight.gotdns.org/prog/neuralpp/network.dtd">'."\n".
		"<!-- Automatically generated by BlackLight's NeuralPerl module -->\n\n".
		"<network name=\"$name\" epochs=\"".$this->{epochs}."\" ".
		"learning_rate=\"".$this->{learning_rate}."\" threshold=\"".
		$this->{threshold}."\">\n".
		"\t<layer class=\"input\" size=\"".$input_size."\"></layer>\n".
		"\t<layer class=\"hidden\" size=\"".$hidden_size."\"></layer>\n".
		"\t<layer class=\"output\" size=\"".$output_size."\"></layer>\n\n";

	for my $i (0..$input_size-1)  {
		for my $j (0..$hidden_size-1)  {
			$xml .=
				"\t<synapsis class=\"inhid\" input=\"$i\" output=\"".
				"$j\" weight=\"".$in_hid_synapses[$i][$j]->{weight}."\">".
				"</synapsis>\n";
		}
	}

	for my $i (0..$hidden_size-1)  {
		for my $j (0..$output_size-1)  {
			$xml .=
				"\t<synapsis class=\"hidout\" input=\"$i\" output=\"".
				"$j\" weight=\"".$hid_out_synapses[$i][$j]->{weight}."\">".
				"</synapsis>\n";
		}
	}

	$xml .= "</network>\n";

	open OUT, "> $file"
		or Carp::croak("Error while writing to $file : $!\n");

	print OUT $xml;
	close OUT;
}

<<EOF;

!!!
!!! XML stuff
!!!

EOF

my $in_network  = 0;
my $in_training = 0;
my $in_input  = 0;
my $in_output = 0;
my @xml_input  = ();
my @xml_expect = ();
my $num_sets  = 0;
my $in_index  = 0;
my $out_index = 0;

sub trainXMLTagStart  {
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

sub trainXMLTagParse  {
	my ($parser, $data) = @_;

	$xml_input [$num_sets][$in_index ++] = $data if $in_input;
	$xml_expect[$num_sets][$out_index++] = $data if $in_output;
}

sub trainXMLTagEnd  {
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
	my @wut = split "\n", $xml;

	for (@wut)  {
		$_ =~ tr/[A-Z]/[a-z]/ if ($_ !~ /<!DOC/i);
	}

	$xml = join "\n", @wut;

	my $parser = new XML::Parser;
	$parser->setHandlers(
		Start => \&trainXMLTagStart,
		Char  => \&trainXMLTagParse,
		End   => \&trainXMLTagEnd
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

my $init_layer = 0;

sub loadXMLTagStart  {
	my ($parser, $name, %attr) = @_;

	for ($name)  {
		if (/^network$/)  {
			$in_network = 1;

			Carp::croak ("Epochs parameter is not specified in XML file\n")
				unless defined $attr{epochs};
			
			Carp::croak ("Learning rate parameter is not specified in XML file\n")
				unless defined $attr{learning_rate};

			$epochs = $attr{epochs};
			$ref_epochs = $epochs;
			$l_rate = $attr{learning_rate};
		} elsif (/^layer$/)  {
			Carp::croak ("Invalid 'layer' tag outside the 'network' tag\n")
				unless $in_network ne 0;

			Carp::croak ("Class parameter not specified\n")
				unless defined $attr{class};
			
			Carp::croak ("Size parameter not specified\n")
				unless defined $attr{size};

			$input_size  = $attr{size} if ($attr{class} =~ /^input$/);
			$hidden_size = $attr{size} if ($attr{class} =~ /^hidden$/);
			$output_size = $attr{size} if ($attr{class} =~ /^output$/);

			if ($input_size ne 0 && $hidden_size ne 0 && $output_size ne 0 && $init_layer eq 0)  {
				for my $i (0..$input_size-1)  {
					my $prop = 0;

					$input_neurons[$i] = {
						propagation_value => $prop,
						activation_value => &$actv($prop),
						threshold => $threshold
					};
				}

				for my $i (0..$hidden_size-1)  {
					my $prop = 0;

					$hidden_neurons[$i] = {
						propagation_value => $prop,
						activation_value => &$actv($prop),
						threshold => $threshold
					};
				}

				for my $i (0..$output_size-1)  {
					my $prop = 0;

					$output_neurons[$i] = {
						propagation_value => $prop,
						activation_value => &$actv($prop),
						threshold => $threshold
					};
				}

				$init_layer = 1;
			}
		} elsif (/^synapsis$/)  {
			Carp::croak ("Invalid 'synapsis' tag outside the 'network' tag\n")
				unless $in_network ne 0;

			Carp::croak ("Class parameter not specified\n")
				unless defined $attr{class};
			
			Carp::croak ("Input parameter not specified\n")
				unless defined $attr{input};
			
			Carp::croak ("Output parameter not specified\n")
				unless defined $attr{output};
			
			Carp::croak ("Weight parameter not specified\n")
				unless defined $attr{weight};

			my $input  = $attr{input};
			my $output = $attr{output};

			if ($attr{class} =~ /^inhid$/)  {
				$in_hid_synapses[$input][$output] = {
					neuron_in => \$input_neurons[$input],
					neuron_out => \$hidden_neurons[$output],
					delta => 0,
					prev_delta => 0,
					weight => $attr{weight}
				};
			} elsif ($attr{class} =~ /^hidout$/)  {
				$hid_out_synapses[$input][$output] = {
					neuron_in => \$hidden_neurons[$input],
					neuron_out => \$output_neurons[$output],
					delta => 0,
					prev_delta => 0,
					weight => $attr{weight}
				};
			}
		} else {
			Carp::croak ("Invalid tag name '$_'\n");
		}
	}
}

sub loadXMLTagEnd  {
	my ($parser, $name) = @_;
	$in_network = 0 if $name =~ /^network$/;
}

1;

