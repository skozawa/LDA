use strict;
use warnings;
use lib 'lib';
use LDA::GibbsSampling;

my $file = $ARGV[0] || "sample.txt";
my $lda = LDA::GibbsSampling->new({ topic_num => 2 });
$lda->initialize($file);
$lda->run;
$lda->show_topic_term_destribution;
$lda->show_doc_topic_distribution;
