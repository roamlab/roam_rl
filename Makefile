SHELL=/bin/bash

.PHONY: setup test-env common default gpu no-avx

default: test-env
	pip uninstall --yes tensorflow tensorflow-gpu
	pip install tensorflow==1.14
	make common

# pre-requisities for garage and baselines, we use garage's well maintained setup script
setup:	
	if [ -d "tmp" ]; then\
	    rm -rf tmp;\
	fi
	mkdir tmp &&\
	    cd tmp &&\
	    git clone git@github.com:rlworkgroup/garage.git &&\
	    echo " " >> mjkey.txt &&\
	    cd garage && ls && ./scripts/setup_linux.sh --mjkey ../mjkey.txt --modify-bashrc &&\
	    cd ../../ && rm -rf tmp

test-env:
	if [ -d "$(VIRTUAL_ENV)" ]; then\
	    echo "found virtual environment";\
	else\
	    echo "virtual environment not found";\
	    exit 1;\
	fi

common:  
	pip install --force-reinstall git+https://git@github.com/openai/baselines@master#egg=baselines 
	pip install --force-reinstall git+https://git@github.com/openai/gym@master#egg=gym 
	pip install git+https://git@github.com/roamlab/confac@master#egg=confac
	pip install -e .

gpu: test-env
	pip uninstall --yes tensorflow tensorflow-gpu
	pip install tensorflow-gpu==1.14
	make common

no-avx: test-env
	pip uninstall --yes tensorflow tensorflow-gpu
	pip install tensorflow==1.5
	make common
