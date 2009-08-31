use ExtUtils::MakeMaker;

# See lib/ExtUtils/MakeMaker.pm for details of how to influence
# the contents of the Makefile that is written.

WriteMakefile(
	'NAME'		=> 'NeuralNet',
	'AUTHOR' 	=> 'BlackLight <blacklight@autistici.org>',
	'ABSTRACT' 	=> 'The Perl module for managing neural networks',
	'LICENSE' 	=> 'GPL',
	'VERSION_FROM'	=> 'NeuralNet.pm',
	
	'MAN3PODS' 	=>  {
		'man/NeuralNet.pod' => 'blib/man3/NeuralNet.3pm'
	},

	'PREREQ_PM'		=> {
		'XML::Parser' => 0
	},
);

