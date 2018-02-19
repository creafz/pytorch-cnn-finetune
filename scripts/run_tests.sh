#!/usr/bin/env bash

# Example usage:
# ./run_tests.sh --python=3.6 --pytorch-source=conda
# ./run_tests.sh --python=3.5 --pytorch-source=github


new_environment=false
delete_environment=false


while [ "$1" != "" ]; do
    PARAM=`echo $1 | awk -F= '{print $1}'`
    VALUE=`echo $1 | awk -F= '{print $2}'`
    case $PARAM in
        --python)
            python_version=$VALUE
            ;;
        --pytorch-source)
            pytorch_source=$VALUE
            ;;
        --new-environment)
            new_environment=true
            ;;
        --delete-environment)
            delete_environment=true
            ;;
        *)
            echo "ERROR: unknown parameter \"$PARAM\""
            exit 1
            ;;
    esac
    shift
done


if ! [[ ${python_version} == 3.4 || ${python_version} == 3.5 || ${python_version} == 3.6 ]] ; then
    echo "ERROR: incorrect python_version"
    exit 1
fi


if ! [[ ${pytorch_source} == conda || ${pytorch_source} == github ]] ; then
    echo "ERROR: incorrect pytorch_source"
    exit 1
fi


environment_name=python_${python_version}_pytorch_${pytorch_source}


if [ ${new_environment} = true ] ; then
    conda remove --name ${environment_name} --all
fi

conda create -y -n ${environment_name} python=${python_version}
source activate ${environment_name}


if python -c "import torch" &> /dev/null; then
    pytorch_installed=true
else
    pytorch_installed=false
fi

if [ ${pytorch_installed} = false ] ; then
    if [ ${pytorch_source} = "conda" ]; then
       conda install -y pytorch torchvision cuda90 -c pytorch
    else
        export CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"
        conda install -y numpy pyyaml mkl setuptools cmake cffi
        conda install -y -c pytorch magma-cuda90
        git clone --recursive https://github.com/pytorch/pytorch
        cd pytorch && python setup.py install
        cd ..
        rm -rf pytorch
    fi
fi


cd ..
pip install .[tests]
py.test


if [ ${delete_environment} = true ] ; then
    conda remove --name ${environment_name} --all
fi
