#!/bin/bash

set -e

outdir=$1
feat_kind=xlsr

[ -z $outdir ] && echo "output directory must be specified" && exit 1
[ ! -z $2 ] && feat_kind=$2

orig_wd="$PWD"

mkdir -p $outdir/wavs

cd $outdir

echo "Downloading data"
[ ! -f librispeech_finetuning.tar.gz ] && wget https://dl.fbaipublicfiles.com/librilight/data/librispeech_finetuning.tgz && mv librispeech_finetuning.tgz librispeech_finetuning.tar.gz

[ ! -f dev-clean.tar.gz ] && wget http://www.openslr.org/resources/12/dev-clean.tar.gz
[ ! -f dev-other.tar.gz ] && wget http://www.openslr.org/resources/12/dev-other.tar.gz
[ ! -f test-clean.tar.gz ] && wget http://www.openslr.org/resources/12/test-clean.tar.gz
[ ! -f test-other.tar.gz ] && wget http://www.openslr.org/resources/12/test-other.tar.gz

echo "Extracting archives and converting to wav"

for split in librispeech_finetuning dev-clean dev-other test-clean test-other; do
    if [ -f wavs/$split/.done ]; then
        continue
    fi
    tar xzf $split.tar.gz
    if [ -d $split ]; then
        flacdir=$split
    elif [ -d LibriSpeech/$split ]; then
        flacdir=LibriSpeech/$split
    else
        echo "flac directory for $split not found, skipping"
        continue
    fi

    mkdir -p wavs/$split
    for f in $(find $flacdir/ -name "*.flac"); do
        ofilename=$(basename $f | sed "s flac wav g")
        flac -c -d -s $f > wavs/$split/$ofilename
    done
    touch wavs/$split/.done
done

echo "Extracting features"
cd $orig_wd
for split in librispeech_finetuning dev-clean dev-other test-clean test-other; do
    if [ -f feats/$feat_kind/$split/.done ]; then
        continue
    fi
    mkdir -p feats/$feat_kind/$split
    python -u golden-retriever/extract_features.py \
        -f $feat_kind \
        $outdir/wavs/$split/ $outdir/feats/$feat_kind/$split/
    touch feats/$feat_kind/$split/.done
done
