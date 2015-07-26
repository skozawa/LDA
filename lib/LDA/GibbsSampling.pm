# LDA with the collapsed Gibbs Sampler
package LDA::GibbsSampling;
use strict;
use warnings;
use utf8;

sub new {
    my ($class, $args) = @_;
    bless {
        # parameter of the Dirichlet prior on the per-document topic distributions
        alpha         => $args->{alpha} || 0.5,
        # parameter of the Dirichlet prior on the per-topic word distribution
        beta          => $args->{beta} || 0.5,
        topic_num     => $args->{topic_num} || 10,
        iteration_num => $args->{iteration_num} || 50,
        # document-term vector [[t11,t12,t13,...],[t21,t22,t23,...]]
        docs              => [],
        # row: document, col: topic, val: topic count in document d
        doc_topic_counts  => [],
        # row: topic, col: term, val: topic term count in all documents
        topic_term_counts => [],
        # topic count in all documents
        topic_counts      => [],
        # row: document, col: term, val: topic
        doc_term_topic    => [],
        # term count in each documents
        term_counts       => [],
        # term index
        term_index        => {},
    }, $class;
}

# read corpus
# - document per line
# - terms are separeted by space
sub read_file {
    my ($self, $file) = @_;
    open(IN, $file) or die "cannot open file $file";
    while (<IN>) {
        chomp;
        push @{$self->{docs}}, [ split / / ];
    }
    close(IN);
}

sub create_term_index {
    my $self = shift;
    my $index = 0;
    foreach my $doc ( @{$self->{docs}} ) {
        foreach my $term ( @$doc ) {
            next if defined $self->{term_index}->{$term};
            $self->{term_index}->{$term} = $index++;
        }
    }
}

# initialize vectors
sub initialize {
    my ($self, $file) = @_;
    $self->read_file($file);
    $self->create_term_index;
    $self->_initialize_doc_term_topic;
    $self->_initialize_doc_topic_counts;
    $self->_initialize_term_counts;
    $self->_initialize_topic_term_counts;
    $self->_initialize_topic_counts;
}

# allocate topic index by random
sub _initialize_doc_term_topic {
    my $self = shift;
    $self->{doc_term_topic} = [ map {
        [ map { int rand($self->{topic_num}) } @$_ ]
    } @{$self->{docs}} ];
}

sub _initialize_doc_topic_counts {
    my $self = shift;
    foreach my $topics ( @{$self->{doc_term_topic}} ) {
        # smoothing by alpha
        my $topic_counts = $self->create_matrix(1, $self->{topic_num}, $self->{alpha});
        $topic_counts->[$_] += 1 for @$topics;
        push @{$self->{doc_topic_counts}}, $topic_counts;
    }
}

sub _initialize_term_counts {
    my $self = shift;
    $self->{term_counts} = $self->create_matrix(
        1, scalar @{$self->{docs}}, $self->{alpha} * $self->{topic_num}
    );
    foreach my $doc_index ( 0 .. $#{$self->{docs}} ) {
        my $term_count = scalar @{$self->{docs}->[$doc_index]};
        $self->{term_counts}->[$doc_index] += $term_count;
    }
}

sub _initialize_topic_term_counts {
    my $self = shift;
    # soomthing by beta
    $self->{topic_term_counts} = $self->create_matrix(
        $self->{topic_num}, scalar keys %{$self->{term_index}}, $self->{beta}
    );
    for my $doc_index ( 0 .. $#{$self->{docs}} ) {
        for my $doc_term_index ( 0 .. $#{$self->{docs}->[$doc_index]} ) {
            my $term = $self->{docs}->[$doc_index]->[$doc_term_index];
            my $term_index = $self->{term_index}->{$term};
            my $topic = $self->{doc_term_topic}->[$doc_index]->[$doc_term_index];
            $self->{topic_term_counts}->[$topic]->[$term_index] += 1;
        }
    }
}

sub _initialize_topic_counts {
    my $self = shift;
    # smoothing by beta
    $self->{topic_counts} = $self->create_matrix(
        1, $self->{topic_num}, $self->{beta} * scalar keys %{$self->{term_index}}
    );

    foreach my $doc_index ( 0 .. $#{$self->{docs}} ) {
        foreach my $term_index ( 0 .. $#{$self->{docs}->[$doc_index]} ) {
            my $topic = $self->{doc_term_topic}->[$doc_index]->[$term_index];
            $self->{topic_counts}->[$topic] += 1;
        }
    }
}

sub create_matrix {
    my ($self, $rows, $cols, $default_value) = @_;
    return [ ($default_value || 0) x $cols ] if $rows == 1;
    return [ map { [ ($default_value || 0) x $cols ] } (1..$rows) ];
}

sub show_topic_term_destribution {
    my $self = shift;

    for my $topic_index ( 0 .. $self->{topic_num} - 1 ) {
        print "topic $topic_index\n";
        foreach my $term ( keys %{$self->{term_index}} ) {
            print "$term: @{[ $self->{topic_term_counts}->[$topic_index]->[$self->{term_index}->{$term}] / $self->{topic_counts}->[$topic_index] ]} ";
        }
        print "\n";
    }
}

sub show_doc_topic_distribution {
    my $self = shift;

    for my $doc_index ( 0 .. $#{$self->{docs}} ) {
        print "document $doc_index\n";
        my $denominator = scalar @{$self->{docs}->[$doc_index]} + $self->{topic_num} * $self->{alpha};
        for my $topic_index ( 0 .. $#{$self->{doc_topic_counts}->[$doc_index]} ) {
            print "topic $topic_index: @{[ $self->{doc_topic_counts}->[$doc_index]->[$topic_index] / $denominator ]} ";
        }
        print "\n";
    }
}


sub run {
    my $self = shift;
    $self->calc for ( 1 .. $self->{iteration_num} );
}

sub calc {
    my $self = shift;

    for my $doc_index ( 0 .. $#{$self->{docs}} ) {
        for my $doc_term_index ( 0 .. $#{$self->{docs}->[$doc_index]} ) {
            my $term = $self->{docs}->[$doc_index]->[$doc_term_index];
            my $term_index = $self->{term_index}->{$term};
            my $topic = $self->{doc_term_topic}->[$doc_index]->[$doc_term_index];

            $self->decrement($doc_index, $topic, $term_index);
            my $new_topic = $self->multinomial_sampling($doc_index, $term_index);
            $self->increment($doc_index, $new_topic, $term_index);

            $self->{doc_term_topic}->[$doc_index]->[$doc_term_index] = $new_topic;
        }
    }
}

sub decrement {
    my ($self, $doc_index, $topic_index, $term_index) = @_;
    $self->{doc_topic_counts}->[$doc_index]->[$topic_index] -= 1;
    $self->{topic_term_counts}->[$topic_index]->[$term_index] -= 1;
    $self->{topic_counts}->[$topic_index] -= 1;
    $self->{term_counts}->[$doc_index] -= 1;
}

sub increment {
    my ($self, $doc_index, $topic_index, $term_index) = @_;
    $self->{doc_topic_counts}->[$doc_index]->[$topic_index] += 1;
    $self->{topic_term_counts}->[$topic_index]->[$term_index] += 1;
    $self->{topic_counts}->[$topic_index] += 1;
    $self->{term_counts}->[$doc_index] += 1;
}

sub multinomial_sampling {
    my ($self, $doc_index, $term_index) = @_;
    my $total_prob = 0;
    my @probs;
    for my $topic_index ( 0 .. $self->{topic_num} - 1 ) {
        my $prob = $self->{doc_topic_counts}->[$doc_index]->[$topic_index] *
            $self->{topic_term_counts}->[$topic_index]->[$term_index] /
                ($self->{topic_counts}->[$topic_index] * $self->{term_counts}->[$doc_index]);
        $total_prob += $prob;
        push @probs, $prob;
    }

    my $rand_prob = rand($total_prob);
    for my $topic_index ( 0 .. $self->{topic_num} - 1 ) {
        $rand_prob -= $probs[$topic_index];
        return $topic_index if $rand_prob < 0;
    }
    return $self->{topic_num} - 1;
}

1;
__END__
